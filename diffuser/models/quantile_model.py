import torch
import torch.nn as nn



class Quantile_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=[256, 256]):
        super().__init__()
        layers = [input_dim, *hidden_units, output_dim]
        layers = list(zip(layers[:-1], layers[1:]))
        self.fc = []
        for layer_idx in range(len(layers)):
            i_dim, o_dim = layers[layer_idx]
            self.fc.append(nn.Linear(i_dim, o_dim))
            if layer_idx == len(layers) - 1:
                pass
            else:
                self.fc.append(nn.LeakyReLU())
        self.fc = nn.Sequential(*self.fc)

    def forward(self, state, return_to_go, timestep):
        return self.fc(torch.cat([state, return_to_go, timestep], dim=-1))


class Quantile_Regression_Model(nn.Module):
    def __init__(self, state_dim, action_dim, quantile_lr, quantile_bins=[0.05, 0.2, 0.3, 0.5, 0.7, 0.8, 0.95], return_to_go_dim=1, timestep_dim=1, device="cuda", hidden_units=[256, 256]):
        super().__init__()
        self.quan_lr = quantile_lr
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.return_to_go_dim = return_to_go_dim
        self.timestep_dim = timestep_dim
        self.hidden_units = hidden_units
        self.quantile_bins = quantile_bins

        self.model_init()

    def model_init(self):
        self.quantile_encoder = Quantile_Encoder(input_dim=self.state_dim+self.return_to_go_dim+self.timestep_dim, output_dim=len(self.quantile_bins), hidden_units=self.hidden_units)
        self.quan_opt = torch.optim.Adam(self.quantile_encoder.parameters(), lr=self.quan_lr)
        self.to(self.device)

    def get_quantile_points(self, state, return_to_go, timestep):
        if len(return_to_go.shape) == 1:
            return_to_go = torch.unsqueeze(return_to_go, dim=-1)
        if len(timestep.shape) == 1:
            timestep = torch.unsqueeze(timestep, dim=-1)
        return self.quantile_encoder(state=state, return_to_go=return_to_go, timestep=timestep)

    def train_model(self, batch):
        states = batch.trajectories[:, 0, self.action_dim:]
        quantile_labels = batch.actionrewards_returns_timesteps[:, 0]
        return_to_go = batch.actionrewards_returns_timesteps[:, 1]
        timestep = batch.actionrewards_returns_timesteps[:, 2]
        y_head = self.get_quantile_points(state=states, return_to_go=torch.unsqueeze(return_to_go, dim=-1), timestep=torch.unsqueeze(timestep, dim=-1))
        quantile_labels = torch.unsqueeze(quantile_labels, dim=-1)
        threshold = torch.zeros(y_head.shape).to(states.device)
        quantile_weights = torch.reshape(torch.tensor(self.quantile_bins), [1, -1]).to(states.device)
        y_sub_y_head = quantile_labels - y_head
        y_head_sub_y = y_head - quantile_labels
        quantile_loss = quantile_weights * torch.where(y_sub_y_head>0, y_sub_y_head, threshold) + (1 - quantile_weights) * torch.where(y_head_sub_y>0, y_head_sub_y, threshold)
        quantile_loss = torch.sum(torch.mean(quantile_loss, dim=-1))
        self.quan_opt.zero_grad()
        quantile_loss.backward()
        self.quan_opt.step()
        return {"quantile_loss": quantile_loss}




























