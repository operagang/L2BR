import torch
import torch.nn as nn
import torch.nn.functional as F
import math
""" https://github.com/wouterkool/attention-learn-to-route
"""


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.inf = 1e9
    def forward(self, Q, K, V, mask):
        d_k = self.d_k        
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d_k) 

        if mask is None:
            mask = torch.zeros_like(attn_score).bool()
        else:
            attn_score = attn_score.masked_fill(mask[:, None, None, :, 0].repeat(1, attn_score.size(1), 1, 1) == True, -self.inf)

        attn_dist = F.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_dist, V)

        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, is_encoder):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.d_k = embed_dim//n_heads
        self.d_v = embed_dim//n_heads
        
        assert self.embed_dim % self.n_heads == 0

        self.is_encoder = is_encoder
        if self.is_encoder:
            self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
            self.layerNorm = nn.LayerNorm(embed_dim, 1e-6)
        self.attention = ScaledDotProductAttention(self.d_k)
    def forward(self, x, mask=None):
        Q,K,V = x
        batchSize, seqLen_Q, seqLen_K = Q.size(0), Q.size(1), K.size(1) 
        if self.is_encoder:
            Q = self.W_Q(Q)
            K = self.W_K(K)
            V = self.W_V(V)
        
        Q = Q.view(batchSize, seqLen_Q, self.n_heads, self.d_k)
        K = K.view(batchSize, seqLen_K, self.n_heads, self.d_k)
        V = V.view(batchSize, seqLen_K, self.n_heads, self.d_v)
        
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batchSize, seqLen_Q, -1)

        if self.is_encoder:
            output = self.W_O(output)
        
        return output

class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, n_heads, embed_dim, ff_hidden, is_encoder=True, init_resweight = 0, resweight_trainable=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.resweight = torch.nn.Parameter(torch.Tensor([init_resweight]), requires_grad = resweight_trainable)
        self.MHA = MultiHeadAttention(n_heads, embed_dim=embed_dim, is_encoder=is_encoder)

        self.FF_sub = nn.Sequential(
                            nn.Linear(embed_dim, ff_hidden),
                            nn.ReLU(),
                            nn.Linear(ff_hidden, embed_dim)
                        )

    def forward(self, x, mask=None):
        t = [x,x,x]
        x = x + self.resweight * self.MHA(t, mask=mask)
        x = x + self.resweight * self.FF_sub(x)
        return x

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        embed_dim = args.embed_dim
        self.embed_dim = embed_dim
        self.n_layers = args.n_encode_layers
        LSTM_num_layers = 1
        self.LSTM_num_layers = LSTM_num_layers
        self.LSTM = nn.LSTM(input_size=embed_dim, hidden_size = embed_dim, batch_first = True, num_layers = LSTM_num_layers)
        self.LSTM_embed = nn.Linear(2*embed_dim, embed_dim, bias=True)
        self.init_positional_encoding = nn.Sequential(
            nn.Linear(1, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 1, bias = True)
        )
        self.init_block_embed = nn.Sequential(
            nn.Linear(2, embed_dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim, bias=True)
        )
        self.encoder_layers = nn.ModuleList([MultiHeadAttentionLayer(args.n_heads, args.embed_dim, args.ff_hidden_dim ) for _ in range(args.n_encode_layers)])

    def forward(self, x, mask=None):
        batch,max_stacks,max_tiers = x.size()
        x = x.clone()
        x = x.view(batch, max_stacks, max_tiers, 1)
        positional_encoding = torch.linspace(0,1,max_tiers).repeat(batch, max_stacks, 1).unsqueeze(-1).to('cuda:0')
        pe = self.init_positional_encoding(positional_encoding)
        x = torch.cat([x, pe], dim=3)

        x = self.init_block_embed(x)
        x = x.view(batch*max_stacks, max_tiers, self.embed_dim)
        output, (hidden_state, _) = self.LSTM(x)

        o = torch.mean(output, dim=1).view(batch, max_stacks, self.embed_dim)

        h = hidden_state[self.LSTM_num_layers-1,:,:].view(batch, max_stacks, self.embed_dim) 

        x = torch.cat([o,h], dim=2)
        x = self.LSTM_embed(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        return (x, torch.mean(x, dim=1))
