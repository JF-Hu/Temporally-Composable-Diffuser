import torch
import torch.nn as nn



class State_Encoder(nn.Module):
    def __init__(self, state_dim, embed_dim, hidden_units=[256, 256]):
        super().__init__()
        input_dim = state_dim
        out_dim = embed_dim
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

class State_Goal_Encoder(nn.Module):
    def __init__(self, state_dim, goal_dim, embed_dim, hidden_units=[256, 256]):
        super().__init__()
        input_dim = state_dim + goal_dim
        out_dim = embed_dim
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

    def forward(self, state, goal):
        return self.fc(torch.cat([state, goal], dim=-1))

class State_to_Goal_Embed_Approximation(nn.Module):
    def __init__(self, state_dim, embed_dim, hidden_units=[256, 256]):
        super().__init__()
        input_dim = state_dim
        out_dim = embed_dim
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


class Bisimulation_Model(nn.Module):
    def __init__(self, state_dim, goal_dim, embed_dim, state_encoder_lr, state_to_goal_lr, state_goal_encoder_lr, gamma, device="cuda", hidden_units=[256, 256]):
        super().__init__()
        self.se_lr = state_encoder_lr
        self.s2g_lr = state_to_goal_lr
        self.sge_lr = state_goal_encoder_lr
        self.device = device
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.goal_dim = goal_dim
        self.gamma = gamma

        self.model_init()

    def model_init(self):
        self.state_encoder = State_Encoder(
            state_dim=self.state_dim, embed_dim=self.embed_dim, hidden_units=self.hidden_units)
        self.s2g_encoder = State_to_Goal_Embed_Approximation(
            state_dim=self.state_dim, embed_dim=self.embed_dim, hidden_units=self.hidden_units)
        self.state_goal_encoder = State_Goal_Encoder(
            state_dim=self.state_dim, goal_dim=self.goal_dim, embed_dim=self.embed_dim, hidden_units=self.hidden_units)
        self.state_encoder_opt = torch.optim.Adam(self.state_encoder.parameters(), lr=self.se_lr)
        self.s2g_encoder_opt = torch.optim.Adam(self.s2g_encoder.parameters(), lr=self.s2g_lr)
        self.state_goal_encoder_opt = torch.optim.Adam(self.state_goal_encoder.parameters(), lr=self.sge_lr)
        self.to(self.device)

    def get_state_embed(self, state):
        return self.state_encoder(state=state)

    def get_state_goal_embed(self, state, goal):
        return self.state_goal_encoder(state=state, goal=goal)

    def get_state_to_goal_embed(self, state):
        return self.s2g_encoder(state=state)

    def train_state_goal_encoder(self, states_i, states_j, rewards_i, rewards_j, next_states_i, next_states_j, goals_i, goals_j):
        with torch.no_grad():
            L_phi_term_3 = torch.norm(
                self.get_state_goal_embed(state=next_states_i, goal=goals_i)-self.get_state_goal_embed(state=next_states_j, goal=goals_j), p=2, dim=-1, keepdim=True)
        L_phi_term_1 = torch.norm(
                self.get_state_goal_embed(state=states_i, goal=goals_i)-self.get_state_goal_embed(state=states_j, goal=goals_j), p=1, dim=-1, keepdim=True)
        L_phi_term_2 = torch.norm(rewards_i-rewards_j, p=2, dim=-1, keepdim=True)
        L_phi = torch.mean(torch.square(L_phi_term_1 - L_phi_term_2 - self.gamma * L_phi_term_3))
        self.state_goal_encoder_opt.zero_grad()
        L_phi.backward()
        self.state_goal_encoder_opt.step()
        return {"state_goal_encoder_loss": L_phi}

    def train_state_encoder(self, states_i, goals_i):
        with torch.no_grad():
            L_psi_term_1 = self.get_state_goal_embed(state=states_i, goal=goals_i) - self.get_state_goal_embed(state=goals_i, goal=goals_i)
        L_psi_term_2 = self.get_state_embed(state=goals_i) - self.get_state_embed(state=states_i)
        L_psi = torch.mean(torch.square(L_psi_term_1 - L_psi_term_2))
        self.state_encoder_opt.zero_grad()
        L_psi.backward()
        self.state_encoder_opt.step()
        return {"state_encoder_loss": L_psi}

    def train_state_to_goal_encoder(self, states_i, goals_i):
        with torch.no_grad():
            target_embed = self.get_state_goal_embed(state=states_i, goal=goals_i)
        pred_embed = self.get_state_to_goal_embed(state=states_i)
        mse_fn = torch.nn.MSELoss()
        L_rho = mse_fn(pred_embed, target_embed)
        self.s2g_encoder_opt.zero_grad()
        L_rho.backward()
        self.s2g_encoder_opt.step()
        return {"state_to_goal_encoder_loss": L_rho}

    def train_model(self, states_i, states_j, rewards_i, rewards_j, next_states_i, next_states_j, goals_i, goals_j):
        loss_info = {}
        state_goal_encoder_loss_info = self.train_state_goal_encoder(
            states_i, states_j, rewards_i, rewards_j, next_states_i, next_states_j, goals_i, goals_j)
        state_encoder_loss_info = self.train_state_encoder(
            states_i, goals_i)
        state_to_goal_encoder_loss_info = self.train_state_to_goal_encoder(
            states_i, goals_i)
        loss_info.update(state_goal_encoder_loss_info)
        loss_info.update(state_encoder_loss_info)
        loss_info.update(state_to_goal_encoder_loss_info)
        return loss_info








