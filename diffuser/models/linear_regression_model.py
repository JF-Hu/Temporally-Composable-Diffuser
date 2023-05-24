import torch
import torch.nn as nn



class Regression_Encoder(nn.Module):
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
                self.fc.append(nn.ReLU())
        self.fc = nn.Sequential(*self.fc)

    def forward(self, state, return_to_go, timestep):
        return self.fc(torch.cat([state, return_to_go, timestep], dim=-1))


class Linear_Regression_Model(nn.Module):
    def __init__(self, state_dim, action_dim, regression_lr, return_to_go_dim=1, timestep_dim=1, device="cuda", hidden_units=[256, 256]):
        super().__init__()
        self.regression_lr = regression_lr
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.return_to_go_dim = return_to_go_dim
        self.timestep_dim = timestep_dim
        self.hidden_units = hidden_units

        self.model_init()

    def model_init(self):
        self.regression_encoder = Regression_Encoder(input_dim=self.state_dim+self.return_to_go_dim+self.timestep_dim, output_dim=1, hidden_units=self.hidden_units)
        self.regression_opt = torch.optim.Adam(self.regression_encoder.parameters(), lr=self.regression_lr)
        self.to(self.device)

    def get_value(self, state, return_to_go, timestep):
        if len(return_to_go.shape) == 1:
            return_to_go = torch.unsqueeze(return_to_go, dim=-1)
        if len(timestep.shape) == 1:
            timestep = torch.unsqueeze(timestep, dim=-1)
        return self.regression_encoder(state=state, return_to_go=return_to_go, timestep=timestep)

    def train_model(self, batch):
        states = batch.trajectories[:, 0, self.action_dim:]
        y_labels = batch.actionrewards_returns_timesteps[:, 0]
        return_to_go = batch.actionrewards_returns_timesteps[:, 1]
        timestep = batch.actionrewards_returns_timesteps[:, 2]
        y_head = self.get_value(state=states, return_to_go=torch.unsqueeze(return_to_go, dim=-1), timestep=torch.unsqueeze(timestep, dim=-1))
        y_labels = y_labels.view(y_head.shape)
        regression_loss = 0.5 * torch.mean(torch.square(y_head - y_labels))
        self.regression_opt.zero_grad()
        regression_loss.backward()
        self.regression_opt.step()
        return {"regression_loss": regression_loss}




























