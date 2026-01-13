import torch
import torch.nn as nn
from encoder import Encoder
from env import Env

class L2BR(nn.Module):

    def __init__(self, args, device):
        super().__init__()

        self.Decoder = Decoder(args, device)
        self.device= device

    def forward(self, x):
        decoder_output = self.Decoder(x)
        return decoder_output


class Decoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.setting = args.problem_type
        self.device = device
        embed_dim = args.embed_dim
        self.embed_dim = embed_dim
        self.concat_embed_dim = self.embed_dim*2
        self.total_embed_dim = self.embed_dim*3
        self.Encoder = Encoder(args).to(device)
        self.Wq_fixed = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.Wk_2 = nn.Linear(embed_dim*3, embed_dim*3, bias=False)

        self.W_O = nn.Sequential(
            nn.Linear(embed_dim*3, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2,embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1)
        ).to(device)

    def compute_logits(self, mask, node_embeddings):
        logits = self.W_O(node_embeddings)
        logtis_with_mask = logits - mask.to(torch.int)*1e9
        return logtis_with_mask.squeeze(dim=2)
    def forward(self, x):
        env = Env(self.device,x)
        encoder_output=self.Encoder(env.x)
        node_embeddings, graph_embedding = encoder_output

        if self.setting == 'rBRP':
            env.find_target_stack()
            target_embeddings = node_embeddings[torch.arange(node_embeddings.size(0)),env.target_stack,:]
            concat_node_embeddings = concat_graph_embedding(target_embeddings, node_embeddings)
            total_embeddings = concat_graph_embedding(graph_embedding, concat_node_embeddings)
            mask = env.create_mask_rBRP()

        elif self.setting == 'uBRP':
            concat_node_embeddings = concat_embedding(node_embeddings, device = self.device)
            total_embeddings = concat_graph_embedding(graph_embedding, concat_node_embeddings)
            mask = env.create_mask_uBRP()

        logits = self.compute_logits(mask, total_embeddings)

        return logits


def concat_embedding(node_embeddings, device='cuda:0'):
    
    batch_size, width, embed_size = node_embeddings.size()

    # Pairwise Concatenation
    reshaped_node_embeddings = node_embeddings.view(batch_size, 1, width, embed_size)
    reshaped_node_embeddings = reshaped_node_embeddings.expand(batch_size, width, width, embed_size)
    newnode_embeddings = torch.cat([reshaped_node_embeddings, reshaped_node_embeddings.transpose(1, 2)], dim=3)

    # Size reformulation
    newnode_embeddings = newnode_embeddings.view(batch_size, width * width, embed_size * 2)
    newnode_embeddings = newnode_embeddings.view(batch_size, width, width, embed_size * 2)
    newnode_embeddings = newnode_embeddings.transpose(1, 2).contiguous().view(batch_size, width * width, embed_size * 2)
    return newnode_embeddings.to(device)
def concat_graph_embedding(graph_embedding, node_embedding, device = 'cuda:0'):
    #Concat the graph embedding (i.e. mean stack embedding)
    batch_size, width, embed_size = node_embedding.size()
    _, embed_size_graph = graph_embedding.size()
    extd_grpah_embedding = graph_embedding.view(batch_size, 1, embed_size_graph).repeat([1, width, 1])
    return torch.cat([extd_grpah_embedding, node_embedding], dim=2).to(device)