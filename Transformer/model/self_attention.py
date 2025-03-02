import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of embedding vector output
            num_heads: Number of self-attention heads
            dropout: Prevents overfitting
        """
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"

        self.embed_dim = embed_dim  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = embed_dim // num_heads  # 512 / 8 = 64

        # Linear layers to project inputs to Q, K, V
        self.query_matrix = nn.Linear(embed_dim, embed_dim)
        self.key_matrix = nn.Linear(embed_dim, embed_dim)
        self.value_matrix = nn.Linear(embed_dim, embed_dim)

        # Final linear layer to combine all heads
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        d_k = Q.shape[-1]  # 64 (head dimension)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Multiply weights with values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear transformations
        Q = self.query_matrix(query)  # (batch, seq_len, embed_dim)
        K = self.key_matrix(key)
        V = self.value_matrix(value)

        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate attention outputs from all heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        # Final linear transformation
        out = self.fc_out(x)

        return out
