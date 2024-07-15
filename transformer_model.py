import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from utils import print_semi,dash

class multi_head(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        embed_dim: int
            the dimension of the input tensor, that is the number of features out of the conv2d layer
        num_heads: int
            the number of heads to split the input tensor into
            !! this should be divisible by 3 !!
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        print_semi("","multi-head attention:",'y')
        print(f' embed_dim: \t{embed_dim}\n num_heads: \t{num_heads}\n head_dim: \t{self.head_dim}')    
        dash()

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
           
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self,x):
        batch_size, seq_length, embed_dim = x.size()

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        # print(f'qkv shape: {qkv.shape}')
        qkv = qkv.permute(2, 0, 3, 1).chunk(3, dim=0) # sperate q,k,v weights
        # print(f'qkv shape after permute: {qkv.shape}')
        q, k, v = map(lambda t: t.squeeze(0), qkv)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        # print(f'q shape: {q.shape} \n k shape: {k.shape} \n v shape: {v.shape} \n attn_scores shape: {attn_scores.shape} \n attn_weights shape: {attn_weights.shape} \n attn_output shape: {attn_output.shape}')
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        
        return attn_output
    

if __name__ == '__main__':
    embed_dim = 516
    num_heads = 3
    model = multi_head(embed_dim, num_heads)
    
    t = torch.rand(4,3,225,225)
    proj = nn.Conv2d(3,embed_dim,3,3)
    x = proj(t)
    flat = x.flatten(2).transpose(1,2)
    print(f'flat shape: {flat.shape}')
    for _ in range(3):
        flat = model(flat)
    print(f'out shape: {flat.shape}')