import torch
#from torch_geometric.data import Data
import FAdo.reex as reex

from utils.utils_data import get_random_nfa, get_final_state_index
from utils.utils_gfa import eliminate_state

class StateEliminationEnvironment:
    def __init__(self, n: int, k: int, d: float, max_n: int, max_k: int, max_regex_len: int) -> None:
        self.n = n
        self.k = k
        self.d = d
        self.max_n = max_n
        self.max_k = max_k
        self.max_regex_len = max_regex_len
        self.n_nodes = max_n + 2
        self.word_to_ix = {'@': 1, '+': 2, '*': 3, '.': 4}
        for i in range(self.max_k):
            self.word_to_ix[str(i)] = i + 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def reset(self):
        while True:
            self.gfa = get_random_nfa(self.n, self.k, self.d, self.max_n)
            if len(self.gfa.States) > 3:
                break
        return self.gfa_to_tensor()
  
    def step(self, action: int):
        eliminate_state(self.gfa, action)
        reward = self.get_reward()
        done = reward != 1
        return self.gfa_to_tensor(), reward, done

    def get_resulting_regex(self) -> reex.RegExp:
        initial_state = self.gfa.Initial
        final_state = get_final_state_index(self.gfa)
        intermediate_state = 3 - (initial_state + final_state)

        alpha = self.gfa.delta[initial_state][intermediate_state]
        beta = reex.CStar(self.gfa.delta[intermediate_state][intermediate_state]) if intermediate_state in self.gfa.delta[intermediate_state] else None
        gamma = self.gfa.delta[intermediate_state][final_state]
        direct_edge = self.gfa.delta[initial_state][final_state] if final_state in self.gfa.delta[initial_state] else None

        result = reex.CConcat(reex.CConcat(alpha, beta), gamma) if beta is not None else reex.CConcat(alpha, gamma)
        result = reex.CDisj(direct_edge, result) if direct_edge is not None else result

        return result

    def get_reward(self) -> int:
        if len(self.gfa.States) == 3:
            result = self.get_resulting_regex()
            length = result.treeLength()
            return length
        elif len(self.gfa.States) == 2:
            initial_state = self.gfa.Initial
            final_state = get_final_state_index(self.gfa)
            length = self.gfa.delta[initial_state][final_state].treeLength()
            return length
        else:
            return 1

    def get_one_hot_vector(self, state_id: int) -> list[int]:
        one_hot_vector = [0] * self.n_nodes
        one_hot_vector[state_id] = 1
        return one_hot_vector

    def get_encoded_regex(self, regex) -> list[int]:
        #NB: This technique cannot be applied to GFA with an alphabet size of more than 10.
        # This is because we replace ' with "" and there is no way to distinguish between 10 and 1, 0.
        encoded_regex = [self.word_to_ix[word] for word in list(regex.rpn().replace("@epsilon", "@").replace("'", ""))[:self.max_regex_len]]
        if len(encoded_regex) < self.max_regex_len:
            encoded_regex = encoded_regex + [0] * (self.max_regex_len - len(encoded_regex))
        assert len(encoded_regex) == self.max_regex_len
        return encoded_regex
    '''
    def gfa_to_graph(self) -> Data:
        x = []
        edge_index = [[], []]
        edge_attr = []
        for source in range(self.n_nodes):
            if source < len(self.gfa.States):
                source_state_number = self.get_one_hot_vector(int(self.gfa.States[source]))
                is_initial_state = 1 if source == self.gfa.Initial else 0
                is_final_state = 1 if source in self.gfa.Final else 0
                out_length = [0] * (self.n_nodes)
                in_length = [0] * (self.n_nodes)
                for target in range(self.n_nodes):
                    if target in self.gfa.delta[source]:
                        edge_index[0].append(source)
                        edge_index[1].append(target)
                        target_id = int(self.gfa.States[target])
                        out_length[target_id] = self.gfa.delta[source][target].treeLength()
                        edge_attr.append(self.get_encoded_regex(self.gfa.delta[source][target]))
                    if target in self.gfa.predecessors[source]:
                        predecessor_id = int(self.gfa.States[target])
                        in_length[predecessor_id] = self.gfa.delta[target][source].treeLength()
                x.append(source_state_number + [is_initial_state, is_final_state] + in_length + out_length)
            else:
                source_state_number = [0] * (self.n_nodes)
                is_initial_state = 0
                is_final_state = 0
                out_length = [0] * (self.n_nodes)
                in_length = [0] * (self.n_nodes)
                x.append(source_state_number + [is_initial_state, is_final_state] + in_length + out_length)
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.LongTensor(edge_attr)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=self.n_nodes)
        return graph
    '''
    def gfa_to_tensor(self) -> dict[torch.FloatTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]:
        nodes = []
        edges = []
        key_padding_mask = [True] * self.n_nodes # for padding
        attn_mask = [[True] * self.n_nodes] * self.n_nodes # for connectivity
        for source in range(self.n_nodes):
            if source < len(self.gfa.States):
                key_padding_mask[source] = False
                source_state_number = self.get_one_hot_vector(int(self.gfa.States[source]))
                is_initial_state = 1 if source == self.gfa.Initial else 0
                is_final_state = 1 if source in self.gfa.Final else 0
                out_length = [0] * (self.n_nodes)
                in_length = [0] * (self.n_nodes)
                out_regex = []
                for target in range(self.n_nodes):
                    if target in self.gfa.delta[source]:
                        attn_mask[source][target] = False
                        target_id = int(self.gfa.States[target])
                        out_length[target_id] = self.gfa.delta[source][target].treeLength()
                        regex = self.get_encoded_regex(self.gfa.delta[source][target])
                    else:
                        regex = [0] * self.max_regex_len
                    out_regex.append(regex)
                    if target in self.gfa.predecessors[source]:
                        attn_mask[target][source] = False
                        predecessor_id = int(self.gfa.States[target])
                        in_length[predecessor_id] = self.gfa.delta[target][source].treeLength()
                nodes.append(source_state_number + [is_initial_state, is_final_state] + in_length + out_length)
                edges.append(out_regex)
            else:
                source_state_number = [0] * (self.n_nodes)
                is_initial_state = 0
                is_final_state = 0
                out_length = [0] * (self.n_nodes)
                in_length = [0] * (self.n_nodes)
                out_regex = [[0] * self.max_regex_len] * self.n_nodes
                nodes.append(source_state_number + [is_initial_state, is_final_state] + in_length + out_length)
                edges.append(out_regex)
        nodes = torch.FloatTensor(nodes).to(self.device).unsqueeze(0)
        edges = torch.LongTensor(edges).to(self.device).unsqueeze(0)
        key_padding_mask = torch.BoolTensor(key_padding_mask).to(self.device).unsqueeze(0)
        attn_mask = torch.BoolTensor(attn_mask).to(self.device).unsqueeze(0)

        observation = {"nodes": nodes, "edges": edges, "key_padding_mask": key_padding_mask, "attn_mask": attn_mask}
        #observation = {"nodes": nodes, "key_padding_mask": key_padding_mask, "attn_mask": attn_mask}
        return observation
