import json
import copy
import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from environment import StateEliminationEnvironment
from network import ReinforceNetwork
from utils.utils import ensure_determinism
from utils.utils_gfa import get_baseline_score

ensure_determinism()

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", action="store_true", help="Enable baseline")
args = parser.parse_args()

cfg = json.load(open('model_configs/reinforce.json'))

hyperparameters = cfg["hyperparameters"]
learning_rate = hyperparameters["learning_rate"]
gamma = hyperparameters["gamma"]

class Reinforce(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = []
        network = cfg["network"]
        self.net = ReinforceNetwork(network).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.net(x)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob, baseline in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * (R - baseline)
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    data = cfg["data"]
    env = StateEliminationEnvironment(data["n"], data["k"], data["d"], data["max_n"], data["max_k"], data["max_regex_len"])
    pi = Reinforce()
    best_score = float("inf")
    score = 0.0
    baseline_score = 0.0

    train = cfg["train"]
    print_interval = train["print_interval"]
    model_path = train["model_path"] + "_baseline" if args.baseline else train["model_path"]

    for n_epi in range(train["n_episodes"]):
        s = env.reset()
        done = False
        c6_length = get_baseline_score(env.gfa)
        baseline = -math.log(c6_length)
        baseline_score += c6_length
        while not done:
            prob = pi(s)
            m = Categorical(prob)
            a = m.sample()
            try:
                s_prime, r, done = env.step(a.item())
                scaled_r = -math.log(r)
            except:
                continue
            baseline = baseline if args.baseline else 0
            pi.put_data((scaled_r, prob.squeeze(0)[a], baseline))
            s = s_prime
            score += r
        pi.train_net()
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_baseline_score = baseline_score / print_interval
            avg_score = score / print_interval
            if avg_score < best_score:
                best_score = avg_score
                torch.save(pi.net.state_dict(), model_path + ".pt")
                with open(model_path + ".log", "w") as f:
                    f.write(f"# of episode: {n_epi}, avg score: {avg_score}, avg baseline score: {avg_baseline_score}")
            print(f"# of episode: {n_epi}, avg score: {avg_score}, avg baseline score: {avg_baseline_score}")
            score = 0.0
            baseline_score = 0.0

if __name__ == '__main__':
    main()
