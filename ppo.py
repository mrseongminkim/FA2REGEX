import json
import copy
import math
import pickle
from pprint import pprint as pp

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

debug = None

class Ppo(nn.Module):
    def __init__(self):
        super().__init__()
        self.data = []
        network = cfg["network"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PpoNetwork(network).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        pi, v = self.net(x)
        return pi, v

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        nodes_lst = []
        #edges_lst = []
        key_padding_mask_lst = []
        attn_mask_lst = []
        prime_nodes_lst = []
        prime_edges_lst = []
        prime_key_padding_mask_lst = []
        prime_attn_mask_lst = []
        for i in range(len(s_lst)):
            observation = s_lst[i]
            #pp(observation)
            prime_observation = s_prime_lst[i]
            #nodes, edges, key_padding_mask, attn_mask = observation["nodes"], observation["edges"], observation["key_padding_mask"], observation["attn_mask"]
            nodes, key_padding_mask, attn_mask = observation["nodes"], observation["key_padding_mask"], observation["attn_mask"]
            #prime_nodes, prime_edges, prime_key_padding_mask, prime_attn_mask = prime_observation["nodes"], prime_observation["edges"], prime_observation["key_padding_mask"], prime_observation["attn_mask"]
            prime_nodes, prime_key_padding_mask, prime_attn_mask = prime_observation["nodes"], prime_observation["key_padding_mask"], prime_observation["attn_mask"]
            nodes_lst.append(nodes)
            #edges_lst.append(edges)
            key_padding_mask_lst.append(key_padding_mask)
            attn_mask_lst.append(attn_mask)
            prime_nodes_lst.append(prime_nodes)
            #prime_edges_lst.append(prime_edges)
            prime_key_padding_mask_lst.append(prime_key_padding_mask)
            prime_attn_mask_lst.append(prime_attn_mask)
        nodes = torch.cat(nodes_lst, dim=0)
        #edges = torch.cat(edges_lst, dim=0)
        key_padding_mask = torch.cat(key_padding_mask_lst, dim=0)
        attn_mask = torch.cat(attn_mask_lst, dim=0)
        prime_nodes = torch.cat(prime_nodes_lst, dim=0)
        #prime_edges = torch.cat(prime_edges_lst, dim=0)
        prime_key_padding_mask = torch.cat(prime_key_padding_mask_lst, dim=0)
        prime_attn_mask = torch.cat(prime_attn_mask_lst, dim=0)

        #s = {"nodes": nodes, "edges": edges, "key_padding_mask": key_padding_mask, "attn_mask": attn_mask}
        #s_prime = {"nodes": prime_nodes, "edges": prime_edges, "key_padding_mask": prime_key_padding_mask, "attn_mask": prime_attn_mask}
        s = {"nodes": nodes, "key_padding_mask": key_padding_mask, "attn_mask": attn_mask}
        #pp(s)
        s_prime = {"nodes": prime_nodes, "key_padding_mask": prime_key_padding_mask, "attn_mask": prime_attn_mask}          
        a = torch.tensor(a_lst).to(self.device)
        r = torch.tensor(r_lst).to(self.device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        prob_a = torch.tensor(prob_a_lst).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        #print(len(self.data))
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        #pp(s)

        for epoch in range(K_epochs):
            with torch.no_grad():
                _, v_prime = self.net(s_prime)
            pi, v = self.net(s)

            debug_pi, debug_v = self.net(debug[0])

            if epoch == 0:
                print("new pi:", pi)
                print("single example:", debug_pi)
                print("prev pi:", debug[1])
                #print("a:", a)
            #exit()

            #print(debug)
            #print(s)
            #exit()

            td_target = r + gamma * v_prime * done_mask
            delta = td_target - v
            delta = delta.to("cpu").detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)

            #pi: torch.Tensor = pi
            #print(pi.shape)

            pi_a = pi.gather(-1, a)
            #print(pi_a.shape)
            #exit()
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v , td_target.detach())

            self.optimizer.zero_grad()
            #print("loss:", loss)
            try:
                loss.mean().backward()
            except:
                #print('pi:', pi[-2])
                #print("pi:", pi)
                #print("a:", a)
                #print("epoch:", epoch)
                #print("pi_a:", pi_a)
                #print("prob_a:", prob_a)
                exit()
            #print("loss:", loss.mean().item())
            self.optimizer.step()

def main():
    data = cfg["data"]
    env = StateEliminationEnvironment(data["n"], data["k"], data["d"], data["max_n"], data["max_k"], data["max_regex_len"])
    model = Ppo()
    best_score = -float("inf")
    score = 0.0
    baseline_score = 0.0

    train = cfg["train"]
    print_interval = train["print_interval"]
    model_path = train["model_path"]

    for n_epi in range(train["n_episodes"]):
        #print("n_epi:", n_epi)
        s = env.reset()
        done = False
        skip_training = False

        gfa = copy.deepcopy(env.gfa)
        bridge_state_name = decompose(gfa)
        c6 = eliminate_by_repeated_state_weight_heuristic(gfa, minimization=False, bridge_state_name=bridge_state_name)
        baseline = c6.treeLength()
        baseline_score += baseline

        #print("New GFA")
        prob_list = []

        while not done:
            for t in range(T_horizon):
                with torch.no_grad():
                    #print(s["nodes"].shape)
                    prob, _ = model(s)
                    print(prob)

                    if len(env.gfa.States) == 5:
                        #print("direct:", prob)
                        single_s = s
                        single_prob = prob

                    #if n_epi == 22:
                    #    pp(s)
                    prob_list.append(prob)
                    #print("states:", env.gfa.States)
                    #print(f"{t} prob:", prob)
                    m = Categorical(prob)
                    a = m.sample().item()
                    #print(a)
                    try:
                        s_prime, r, done = env.step(a)
                    except:
                        gfa = env.gfa
                        with open(f"error_gfa_{n_epi}.log", "w"):
                            f.write(f"states: {gfa.States}\n")
                            f.write(f"initial: {gfa.Initial}\n")
                            f.write(f"final: {gfa.Final}\n")
                            f.write(f"delta: {gfa.delta}\n")
                            f.write(f"predecessors: {gfa.predecessors}\n")
                            f.write(f"action: {a.item()}\n")
                        pickle.dump(gfa, open(f"error_gfa_{n_epi}.pkl", "wb"))
                        skip_training = True
                        break
                    scaled_r = -math.log(r)
                    #print(scaled_r)
                    model.put_data((s, a, scaled_r, s_prime, prob.squeeze(0)[a].item(), done))
                    s = s_prime

                    score += r
                    if done:
                        break
        if skip_training:
            break
        global debug
        debug = (single_s, single_prob)
        #debug = torch.cat(prob_list, dim=0)
        model.train_net()
        #print(model.net.embedding_with_lstm.embed.weight)

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
