import torch
import torch.nn as nn

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Value_Net(nn.Module):
    def __init__(self, state_dim, hidden_units=[256, 256]):
        super().__init__()
        input_dim = state_dim
        out_dim = 1
        layers = [input_dim, *hidden_units, out_dim]
        layers = list(zip(layers[:-1], layers[1:]))
        self.fc = []
        for layer_idx in range(len(layers)):
            i_dim, o_dim = layers[layer_idx]
            self.fc.append(nn.Linear(i_dim, o_dim))
            if layer_idx == len(layers) - 1:
                pass
            else:
                self.fc.append(nn.ReLU())
        self.fc = nn.Sequential(*self.fc)

    def forward(self, state):
        return self.fc(state)

class Value_Function(nn.Module):
    def __init__(self, state_dim, v_lr, device="cuda", gamma=0.99, ema_decay=0.995, step_start_ema=2000, update_ema_every=10, hidden_units=[256, 256]):
        super().__init__()
        self.v_lr = v_lr
        self.device = device
        self.state_dim = state_dim
        self.hidden_units = hidden_units
        self.gamma = gamma
        self.ema = EMA(ema_decay)
        self.step = 0
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.model_init()

    def model_init(self):
        self.v_net = Value_Net(
            state_dim=self.state_dim, hidden_units=self.hidden_units)
        self.target_v_net = Value_Net(
            state_dim=self.state_dim, hidden_units=self.hidden_units)
        self.v_opt = torch.optim.Adam(self.v_net.parameters(), lr=self.v_lr)
        self.to(self.device)

    def reset_parameters(self):
        self.target_v_net.load_state_dict(self.v_net.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.target_v_net, self.v_net)

    def get_state_value(self, state):
        return self.v_net(state=state)

    def train_state_value_function(self, states, rewards, next_states):
        with torch.no_grad():
            target_v = self.get_state_value(state=next_states)
        current_v = self.get_state_value(state=states)
        v_loss = 0.5 * torch.square(rewards + self.gamma * target_v - current_v)
        v_loss = v_loss.mean()
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        self.step += 1

        if self.step % self.update_ema_every == 0:
            self.step_ema()

        return {"v_loss": v_loss}

    def train_model(self, states, rewards, next_states):
        loss_info = {}
        v_loss_info = self.train_state_value_function(states, rewards, next_states)
        loss_info.update(v_loss_info)
        return loss_info








