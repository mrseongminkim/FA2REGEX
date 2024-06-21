import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from environment import StateEliminationEnvironment
from network import PpoNetwork
from utils.utils import ensure_determinism
from utils.utils_gfa import get_baseline_score

#torch.autograd.set_detect_anomaly(True)
#torch.set_printoptions(precision=4, linewidth=200, sci_mode=False, threshold=10_000)

ensure_determinism()

cfg = json.load(open('model_configs/ppo.json'))

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
    
    def batch_states(self):
        with torch.no_grad():
            nodes_lst = []
            key_padding_mask_lst = []
            attn_mask_lst = []
            for state in self.states:
                nodes, key_padding_mask, attn_mask = state["nodes"], state["key_padding_mask"], state["attn_mask"]
                nodes_lst.append(nodes)
                key_padding_mask_lst.append(key_padding_mask)
                attn_mask_lst.append(attn_mask)
            nodes = torch.cat(nodes_lst, dim=0)
            key_padding_mask = torch.cat(key_padding_mask_lst, dim=0)
            attn_mask = torch.cat(attn_mask_lst, dim=0)
            states = {"nodes": nodes, "key_padding_mask": key_padding_mask, "attn_mask": attn_mask}
            return states

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]

class Ppo:
    def __init__(self, hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = hyperparameters["learning_rate"]
        self.gamma = hyperparameters["gamma"]
        self.eps_clip = hyperparameters["eps_clip"]
        self.K_epochs = hyperparameters["K_epochs"]
        self.buffer = RolloutBuffer()
        self.net = PpoNetwork(cfg["network"]).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        self.criterion = nn.MSELoss()
    
    def act(self, state):
        with torch.no_grad():
            pi, v = self.net(state)
            pi = Categorical(pi)
            action = pi.sample()
            log_prob = pi.log_prob(action)
            return action, log_prob, v
    
    def evaluate(self, state, action):
        pi, v = self.net(state)
        pi = Categorical(pi)
        log_prob = pi.log_prob(action)
        entropy = pi.entropy()
        return log_prob, v, entropy

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, state_value = self.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(log_prob)
            self.buffer.state_values.append(state_value)
            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Normalizing the rewards
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = self.buffer.batch_states()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_log_probs = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach() 

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            log_probs, state_values, entropy = self.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.criterion(state_values, rewards) - 0.01 * entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # clear buffer
        self.buffer.clear()

def main():
    data = cfg["data"]
    env = StateEliminationEnvironment(data["n"], data["k"], data["d"], data["max_n"], data["max_k"], data["max_regex_len"])
    hyperparameters = cfg["hyperparameters"]
    agent = Ppo(hyperparameters)
    best_score = float("inf")
    score = 0.0
    baseline_score = 0.0

    train = cfg["train"]
    print_interval = train["print_interval"]
    model_path = train["model_path"]

    for n_epi in range(train["n_episodes"]):
        state = env.reset()
        done = False
        baseline_score += get_baseline_score(env.gfa)
        while not done:
            action = agent.select_action(state)
            state, r, done = env.step(action)
            score += r
            scaled_r = -math.log(r)
            agent.buffer.rewards.append(scaled_r)
            agent.buffer.dones.append(done)
        # episodic update
        agent.update()
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_baseline_score = baseline_score / print_interval
            avg_score = score / print_interval
            agent.scheduler.step(avg_score)
            if avg_score < best_score:
                best_score = avg_score
                torch.save(agent.net.state_dict(), model_path + ".pt")
                with open(model_path + ".log", "w") as f:
                    f.write(f"# of episode: {n_epi}, avg score: {avg_score}, avg baseline score: {avg_baseline_score}")
            print(f"# of episode: {n_epi}, avg score: {avg_score:10.2f}, avg baseline score: {avg_baseline_score:5.2f}")
            score = 0.0
            baseline_score = 0.0

if __name__ == '__main__':
    main()
