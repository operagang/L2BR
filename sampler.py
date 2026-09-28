import torch.nn as nn 
import torch
class Sampler(nn.Module):
    def __init__(self, n_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples


class TopKSampler(Sampler):
    def forward(self, logits):
        return torch.topk(logits, self.n_samples, dim=1)[1]


class CategoricalSampler(Sampler):
    def forward(self, logits):
        return torch.multinomial(logits, self.n_samples)

class New_Sampler(Sampler):
    def __init__(self, T = 1, **kwargs):
        super().__init__(**kwargs)
        self.T = T
    def forward(self, logits):
        new_logits = logits/self.T
        p_logits = torch.softmax(new_logits, dim=1)
        return torch.multinomial(p_logits, self.n_samples)
