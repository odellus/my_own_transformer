import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_weights = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

    def forward(self, input_embeds):
        # input_embeds is a seq_length X embed_dim matrix.
        query, key, value = self.attn_weights(input_embeds).split(self.embed_dim, dim=-1)
        w = nn.Softmax(dim=-1)(torch.matmul(query, key.transpose(-2,-1)))
        return torch.matmul(w, value)

class MLP(nn.Module):
    def __init__(self, embed_dim, intermediate_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        if intermediate_dim is None:
            intermediate_dim = 4 * embed_dim
        self.intermediate_dim = intermediate_dim
        
        self.fc1 = nn.Linear(self.embed_dim, self.intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_block = SelfAttention(embed_dim)
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
        hidden_states = ff_hidden_states + residual # One more time
        return hidden_states
