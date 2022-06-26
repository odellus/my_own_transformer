import time
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, reduce

class SelfAttention(nn.Module):
    '''Classic self-attention
    '''
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.one_over_sqrt_embed_dim = 1./np.sqrt(embed_dim)
        self.attn_weights = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_states):
        # input_embeds is a seq_length X embed_dim matrix.
        query, key, value = self.attn_weights(hidden_states).split(self.embed_dim, dim=-1)
        w = self.softmax(
            torch.matmul(
                query, 
                key.transpose(-2,-1) * self.one_over_sqrt_embed_dim,
            )
        ) 
        return torch.matmul(w, value)

class CrossAttention(nn.Module):
    '''Cross attention module for encoder-decoder architectures
    '''
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.one_over_sqrt_embed_dim = 1./np.sqrt(embed_dim)
        self.self_attn_weights = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.cross_attn_weights = nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, hidden_states, decoder_hidden_states):
        query = self.self_attn_weights(hidden_states)
        key, value = self.cross_attn_weights(decoder_hidden_states).split(self.embed_dim, dim=-1)
        w = self.softmax(
            torch.matmul(
                query,
                key.transpose(-2,-1) * self.one_over_sqrt_embed_dim
            )
        )
        return torch.matmul(w, value)


class MultiHeadedAttention(nn.Module):
    '''
    '''
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0
        # We need to go from [batch_size, seq_length, embed_dim]
        # to [batch_size, num_heads, seq_length, head_dim]
        self.head_dim = embed_dim // num_heads
        self.attn_weights = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.inv_sqrt_dim = 1. / np.sqrt(self.embed_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def split_heads(self, hidden_states):
        return rearrange(hidden_states, 'b s (h d) -> b h s d', h=self.num_heads)

    def merge_heads(self, hidden_layer):
        return rearrange(hidden_layer, 'b h s d -> b s (h d)', h=self.num_heads)

    def forward(self, hidden_states):
        query, key, value = self.attn_weights(hidden_states).split(self.embed_dim, dim=-1)
        query_layer = self.split_heads(query)
        key_layer = self.split_heads(key)
        value_layer = self.split_heads(value)
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))
        attn_scores *= self.inv_sqrt_dim # normalize
        w = self.softmax(attn_scores)
        outputs = torch.matmul(w, value_layer)
        return self.merge_heads(outputs)
    
class MLP(nn.Module):
    '''
    '''
    def __init__(self, embed_dim, intermediate_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        if intermediate_dim is None:
            intermediate_dim = 4 * self.embed_dim
        self.intermediate_dim = intermediate_dim
        
        self.fc1 = nn.Linear(self.embed_dim, self.intermediate_dim)
        self.fc2 = nn.Linear(self.intermediate_dim, self.embed_dim)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class TransformerBlock(nn.Module):
    '''
    '''
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn_block = MultiHeadedAttention(embed_dim, num_heads)
        self.feed_forward = MLP(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim) # Do we want this or weight sharing?

    def forward(self, hidden_states):
        # hidden_states is a seq_length X embed_dim matrix.
        # This is a toy, so no batch dimension.
        residual = hidden_states # Set up skip connection.
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attn_block(hidden_states)
        hidden_states += residual # Add residual
        residual = hidden_states # Do it again.
        hidden_states = self.ln2(hidden_states)
        ff_hidden_states = self.feed_forward(hidden_states)
        hidden_states = ff_hidden_states + residual # One more time.
        return hidden_states

def main():
    '''
    '''
    batch_size = 4
    seq_length = 2048
    embed_dim = 1024
    num_heads = 8
    # Let's make a single input embedding sample with no batch dimension.
    inputs = torch.randn(batch_size, seq_length, embed_dim)
    inputs = inputs.to('cuda')
    transformer = TransformerBlock(embed_dim, num_heads)
    transformer = transformer.to('cuda')
    t = time.time()
    outputs = transformer(inputs)
    print(f'{time.time() - t}')
    print(outputs.shape) 

if __name__ == "__main__":
    main()