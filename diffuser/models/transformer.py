
from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from diffuser.models.helpers import (
    SinusoidalPosEmb,
)
from torch.distributions import Bernoulli

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if len(freqs_cis.shape) == 3:
        shape = [d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, n_heads, dim, max_batch_size, max_seq_len):
        super().__init__()

        self.head_dim = dim // n_heads
        self.n_heads = n_heads

        self.wq = nn.Linear(in_features=dim, out_features=n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(in_features=dim, out_features=n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(in_features=dim, out_features=n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(in_features=dim, out_features=n_heads * self.head_dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor], start_pos=0):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(in_features=dim, out_features=hidden_dim, bias=False)
        self.w2 = nn.Linear(in_features=hidden_dim, out_features=dim, bias=False)
        self.w3 = nn.Linear(in_features=dim, out_features=hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, norm_eps, dim, n_heads=8, max_batch_size=32, max_seq_len=100, multiple_of=256):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(n_heads, dim, max_batch_size, max_seq_len)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis=freqs_cis, mask=mask, start_pos=start_pos)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, n_layers, sequen_dim, calc_energy=False, n_heads=8, dim=512,
                 max_seq_len=1000, slide_seq_len=100, norm_eps=1e-5, return_type=12,
                 returns_condition=False,
                 goals_condition=False,
                 condition_dropout=0.1,
                 ):
        super().__init__()
        self.n_layers = n_layers
        self.sequen_dim = sequen_dim
        self.calc_energy = calc_energy
        self.n_heads = n_heads
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.slide_seq_len = slide_seq_len
        self.norm_eps = norm_eps
        self.return_type = return_type
        self.returns_condition = returns_condition
        self.goals_condition = goals_condition
        self.condition_dropout = condition_dropout

        if calc_energy:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.Mish()

        time_dim = dim // n_heads
        if self.return_type == 4:
            time_outdim = dim
            return_outdim = dim
        else:
            raise Exception("self.return_type is wrong !!!")
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            act_fn,
            nn.Linear(time_dim * 4, time_outdim),
        )

        returns_mlp_input_dim = 2
        self.returns_mlp = nn.Sequential(
            nn.Linear(returns_mlp_input_dim, time_dim),
            act_fn,
            nn.Linear(time_dim, time_dim * 4),
            act_fn,
            nn.Linear(time_dim * 4, return_outdim),
        )
        self.mask_dist = Bernoulli(probs=1 - self.condition_dropout)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(TransformerBlock(layer_id=layer_id, norm_eps=norm_eps, dim=dim, n_heads=n_heads, max_seq_len=slide_seq_len))

        self.norm = RMSNorm(dim, eps=norm_eps)

        if self.return_type == 4:  # todo use the total generative sequence
            self.output = nn.Linear(in_features=dim, out_features=sequen_dim, bias=False)
        else:
            raise Exception("The noise net is wrong !!!")

        self.freqs_cis = precompute_freqs_cis(
            dim // n_heads, max_seq_len * 2
        )

        self.input_dim_mapping = nn.Linear(sequen_dim, out_features=dim, bias=False)


    def forward(self, x: torch.Tensor, cond, time: torch.Tensor, returns=None, start_pos=0, goals=None, use_dropout=True, force_dropout=False):
        _bsz, seqlen, _ = x.shape
        # h = self.tok_embeddings(tokens)
        h = self.input_dim_mapping(x)

        if self.return_type == 4:
            t = self.time_mlp(time)
            h = h + torch.unsqueeze(t, dim=1)
            returns_embed = self.returns_mlp(returns)
            h = h + torch.unsqueeze(returns_embed, dim=1)
        else:
            raise Exception("self.return_type is wrong !!!")

        self.freqs_cis = self.freqs_cis.to(h.device)

        if self.return_type == 4:  # todo use the total generative sequence
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
            mask = None
            if seqlen > 1:
                mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=x.device)
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        else:
            raise Exception("The noise net is wrong !!!")

        for layer in self.layers:
            h = layer(x=h, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
        h = self.norm(h)
        if self.return_type == 4: # todo use the total generative sequence
            output = self.output(h[:, :, :])
        else:
            raise Exception("The noise net is wrong !!!")
        return output.float()

    @torch.inference_mode()
    def get_pred(self, x: torch.Tensor, cond, time: torch.Tensor, start_pos=0, returns=None, goals=None, use_dropout=True, force_dropout=False):

        _bsz, seqlen, _ = x.shape
        h = self.input_dim_mapping(x)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, :, :])  # only compute last logits
        return output.float()





