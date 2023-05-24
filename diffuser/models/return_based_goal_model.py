import torch
import torch.nn as nn


class State_to_Goal_Approximation(nn.Module):
    def __init__(self, state_dim, goal_dim, hidden_units=[256, 256]):
        super().__init__()
        input_dim = state_dim
        out_dim = goal_dim
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

    def forward(self, state, reward):
        # return self.fc(torch.cat([state, reward], dim=-1))
        return self.fc(state)


class Return_based_State_to_Goal_Model(nn.Module):
    def __init__(self, state_dim, goal_dim, state_to_goal_lr, device="cuda", hidden_units=[256, 256]):
        super().__init__()
        self.s2g_lr = state_to_goal_lr
        self.device = device
        self.state_dim = state_dim
        self.hidden_units = hidden_units
        self.goal_dim = goal_dim

        self.model_init()

    def model_init(self):
        self.s2g_encoder = State_to_Goal_Approximation(
            state_dim=self.state_dim, goal_dim=self.goal_dim, hidden_units=self.hidden_units)
        self.s2g_opt = torch.optim.Adam(self.s2g_encoder.parameters(), lr=self.s2g_lr)
        self.to(self.device)

    def get_state_to_goal(self, state, reward):
        return self.s2g_encoder(state=state, reward=reward)

    def train_state_to_goal_encoder(self, states_i, goals_i, rewards):
        pred_embed = self.get_state_to_goal(state=states_i, reward=rewards)
        L_rho = torch.mean(0.5 * torch.sum(torch.square(pred_embed - goals_i), dim=-1))
        # mse_fn = torch.nn.MSELoss()
        # L_rho = mse_fn(pred_embed, goals_i)
        self.s2g_opt.zero_grad()
        L_rho.backward()
        self.s2g_opt.step()
        return {"state_to_goal_encoder_loss": L_rho}

    def train_model(self, states_i, rewards_i, goals_i):
        states_i = states_i.type(torch.float32)
        goals_i = goals_i.type(torch.float32)
        rewards_i = rewards_i.type(torch.float32)
        loss_info = {}
        state_to_goal_encoder_loss_info = self.train_state_to_goal_encoder(
            states_i, goals_i, rewards_i)
        loss_info.update(state_to_goal_encoder_loss_info)
        return loss_info








