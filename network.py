import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingWithLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim, batch_first=True, bidirectional=True)

    def forward(self, regex):
        regex = self.embed(regex)
        regex, _ = self.lstm(regex)
        regex = regex[:, -1]
        return regex

class MAB(nn.Module):
    def __init__(self, embed_dim, num_heads, ln=False):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        if ln:
            self.ln0 = nn.LayerNorm(embed_dim)
            self.ln1 = nn.LayerNorm(embed_dim)
        self.rff = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, y, key_padding_mask=None, attn_mask=None):
        attn_output, attn_output_weights = self.multihead_attention(x, y, y, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = x + attn_output
        h = h if getattr(self, 'ln0', None) is None else self.ln0(h)
        h = h + F.relu(self.rff(h))
        h = h if getattr(self, 'ln1', None) is None else self.ln1(h)
        return h

class SAB(nn.Module):
    def __init__(self, embed_dim, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(embed_dim, num_heads, ln=ln)
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        return self.mab(x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

class PMA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, embed_dim))
        self.mab = MAB(embed_dim, num_heads, ln=ln)
    
    def forward(self, x, key_padding_mask=None):
        return self.mab(self.seed_vectors.repeat(x.size(0), 1, 1), x, key_padding_mask=key_padding_mask)

class SIT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_seeds, n_layers, ln=False):
        super().__init__()
        self.set_attention_layers = nn.ModuleList([SAB(embed_dim, num_heads, ln=ln) for _ in range(n_layers - 1)])
        self.set_pooling = PMA(embed_dim, num_heads, num_seeds, ln=ln)

    def forward(self, S, E, key_padding_mask=None, attn_mask=None):
        # concat
        Z = torch.cat([S, E], dim=1)

        # attend layers
        for layer in self.set_attention_layers:
            Z = layer(Z, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # get new S
        S = self.set_pooling(Z, key_padding_mask=key_padding_mask)

        # extract new E, preventing cuda mistakes
        indices = torch.tensor([e + 1 for e in range(Z.size()[1] - 1)])
        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
            indices = torch.tensor([e + 1 for e in range(
                Z.size()[1] - 1)]).cuda()
        E = torch.index_select(Z, dim=1, index=indices)

        return S, E

class ReinforceNetwork(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        #self.embedding_with_lstm = EmbeddingWithLSTM(cfg["vocab_size"], cfg["embedding_dim"], cfg["lstm_dim"])
        #self.regex_pooling = PMA(cfg["lstm_dim"] * 2, 1, 1, ln=False)
        self.embed = nn.Linear(38, 256)
        self.encoder = nn.ModuleList([SAB(embed_dim=256, num_heads=1, ln=False) for _ in range(2)])
        self.decoder = PMA(embed_dim=256, num_heads=1, num_seeds=1, ln=False)
        self.sit = SIT(embed_dim=256, num_heads=1, num_seeds=1, n_layers=2, ln=False)
        '''
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        '''
    
    def forward(self, observation: tuple[torch.FloatTensor, torch.BoolTensor, torch.BoolTensor]):
        nodes, key_padding_mask, attn_mask = observation["nodes"], observation["key_padding_mask"], observation["attn_mask"]

        n_batches = nodes.size(0)
        n_nodes = nodes.size(1)
        
        #edges = edges.view(n_batches * n_nodes * n_nodes, max_regex_len)
        #edges = self.embedding_with_lstm(edges)
        #edges = edges.view(n_batches, n_nodes, n_nodes, -1)

        #out_edges = edges.view(n_batches * n_nodes, n_nodes, -1)
        #out_transition = self.regex_pooling(out_edges).squeeze(1)
        #out_transition = out_transition.view(n_batches, n_nodes, -1)

        #in_edges = edges.permute(0, 2, 1, 3).reshape(n_batches * n_nodes, n_nodes, -1)
        #in_transition = self.regex_pooling(in_edges).squeeze(1)
        #in_transition = in_transition.view(n_batches, n_nodes, -1)

        #nodes = torch.cat((nodes, in_transition, out_transition), dim=-1)
        nodes = F.relu(self.embed(nodes))

        for encoder in self.encoder:
            nodes = encoder(nodes, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        set_representation = self.decoder(nodes, key_padding_mask=key_padding_mask)

        set_key_padding_mask = torch.full((n_batches, n_nodes + 1), False, device=key_padding_mask.device)
        set_key_padding_mask[:, :n_nodes] = key_padding_mask
        set_attn_mask = torch.full((n_batches, n_nodes + 1, n_nodes + 1), False, device=attn_mask.device)
        set_attn_mask[:, :n_nodes, :n_nodes] = attn_mask

        set_representation, nodes = self.sit(set_representation, nodes, set_key_padding_mask, set_attn_mask)

        output_mask = key_padding_mask
        output_mask[:, 0:2] = True
        attention = torch.bmm(set_representation, nodes.permute(0, 2, 1))
        attention = attention.masked_fill(output_mask, -float("inf"))
        attention = F.softmax(attention, dim=-1).squeeze(1)
        return attention

class PpoNetwork(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.embedding_with_lstm = EmbeddingWithLSTM(cfg["vocab_size"], cfg["embedding_dim"], cfg["lstm_dim"])
        self.regex_pooling = PMA(cfg["lstm_dim"] * 2, 1, 1, ln=False)
        self.embed = nn.Linear(38, 256)
        self.encoder = nn.ModuleList([SAB(embed_dim=256, num_heads=1, ln=False) for _ in range(2)])
        self.decoder = PMA(embed_dim=256, num_heads=1, num_seeds=1, ln=False)
        self.sit = SIT(embed_dim=256, num_heads=1, num_seeds=1, n_layers=2, ln=False)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        def nan_hook(self, inp, output):
            #print(self.__class__.__name__)
            #print(inp)
            #print(output.shape)
            #exit()
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            for i, out in enumerate(outputs):
                if isinstance(out, tuple):
                    out = out[0]
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    print("weight:", self.weight)
                    exit()
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

        #for submodule in self.modules():
        #    submodule.register_forward_hook(nan_hook)

    def forward(self, observation: tuple[torch.FloatTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]):
        #nodes, edges, key_padding_mask, attn_mask = observation["nodes"], observation["edges"], observation["key_padding_mask"], observation["attn_mask"]
        nodes, key_padding_mask, attn_mask = observation["nodes"], observation["key_padding_mask"], observation["attn_mask"]

        n_batches = nodes.size(0)
        n_nodes = nodes.size(1)
        '''
        n_batches = edges.size(0)
        n_nodes = edges.size(1)
        max_regex_len = edges.size(-1)
        
        edges = edges.view(n_batches * n_nodes * n_nodes, max_regex_len)
        edges = self.embedding_with_lstm(edges)
        edges = edges.view(n_batches, n_nodes, n_nodes, -1)

        out_edges = edges.view(n_batches * n_nodes, n_nodes, -1)
        out_transition = self.regex_pooling(out_edges).squeeze(1)
        out_transition = out_transition.view(n_batches, n_nodes, -1)

        in_edges = edges.permute(0, 2, 1, 3).reshape(n_batches * n_nodes, n_nodes, -1)
        in_transition = self.regex_pooling(in_edges).squeeze(1)
        in_transition = in_transition.view(n_batches, n_nodes, -1)

        nodes = torch.cat((nodes, in_transition, out_transition), dim=-1)
        '''
        nodes = F.relu(self.embed(nodes))

        for encoder in self.encoder:
            nodes = encoder(nodes, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        set_representation = self.decoder(nodes, key_padding_mask=key_padding_mask)

        set_key_padding_mask = torch.full((n_batches, n_nodes + 1), False, device=key_padding_mask.device)
        set_key_padding_mask[:, :n_nodes] = key_padding_mask
        set_attn_mask = torch.full((n_batches, n_nodes + 1, n_nodes + 1), False, device=attn_mask.device)
        set_attn_mask[:, :n_nodes, :n_nodes] = attn_mask

        set_representation, nodes = self.sit(set_representation, nodes, set_key_padding_mask, set_attn_mask)

        value = self.value_head(set_representation).squeeze(-1)
        #print("before linear:", nodes)
        nodes = self.policy_head(nodes).squeeze(-1)

        output_mask = key_padding_mask
        output_mask[:, 0:2] = True

        #print("mask:", output_mask)
        #print("after linear:", nodes)

        nodes = nodes.masked_fill(output_mask, -1e9)
        prob = F.softmax(nodes, dim=-1)
        
        return prob, value
