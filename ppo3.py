import json
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from environment import StateEliminationEnvironment
from network import PpoNetwork
from utils.utils import ensure_determinism
from utils.utils_gfa import decompose, eliminate_by_repeated_state_weight_heuristic

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(precision=4, linewidth=200, sci_mode=False, threshold=10_000)

ensure_determinism()

cfg = json.load(open('model_configs/ppo.json'))

hyperparameters = cfg["hyperparameters"]
learning_rate = hyperparameters["learning_rate"]
gamma = hyperparameters["gamma"]
lmbda = hyperparameters["lmbda"]
eps_clip = hyperparameters["eps_clip"]
K_epochs = hyperparameters["K_epochs"]
T_horizon = hyperparameters["T_horizon"]

def main():
    data = cfg["data"]
    env = StateEliminationEnvironment(data["n"], data["k"], data["d"], data["max_n"], data["max_k"], data["max_regex_len"])
    model = Ppo()
    best_score = float("inf")
    score = 0.0
    baseline_score = 0.0

    train = cfg["train"]
    print_interval = train["print_interval"]
    model_path = train["model_path"]

    for n_epi in range(train["n_episodes"]):
        s = env.reset()
        done = False
        baseline_score += get_baseline_score(env.gfa)
        while not done:
            action = model.select_action(s)
            state, r, done = env.step(action)
            score += r
            scaled_r = -math.log(r)
            model.buffer.rewards.append(r)
            model.buffer.is_terminals.append(done)
        # episodic update
        agent.update()
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_baseline_score = baseline_score / print_interval
            avg_score = score / print_interval
            if avg_score > best_score:
                best_score = avg_score
                torch.save(model.net.state_dict(), model_path + ".pt")
                with open(model_path + ".log", "w") as f:
                    f.write(f"# of episode: {n_epi}, avg score: {avg_score}, avg baseline score: {avg_baseline_score}")
            print(f"# of episode: {n_epi}, avg score: {avg_score}, avg baseline score: {avg_baseline_score}")
            score = 0.0
            baseline_score = 0.0

if __name__ == '__main__':
    main()
