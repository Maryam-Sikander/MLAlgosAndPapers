import torch
import torch.nn as nn
from model.self_attention import MultiHeadAttention
from Transformer.model.embedding_layers import Norm, InputEmbeddings, PositionalEncoding

class EncoderLayer(nn.Module):
  def __init__(self, embed_dim, num_heads, dff=2048, dropout=0.1):
    """
    Args:
      embed_dim: dimensionality of the embedding vector
      num_heads: number of attention heads (8)
      dff: dimensionality of feed-forward network (2048)
      dropout: prevent overfitting
    """
    super(EncoderLayer, self).__init__()
    self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
    self.feed_forward = nn.Sequential(
        nn.Linear(embed_dim, dff), # 512, 2048
        nn.ReLU(),
        nn.Linear(dff, embed_dim)
    )
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    self.norm1 = Norm(embed_dim)
    self.norm2 = Norm(embed_dim)

  def forward(self, src, mask=None):
    """
    Args:
      src:  input sequence (11,32,512)
    """
    x1 = self.norm1(src)
    # Add and multihead attention
    x2 = src + self.dropout1(self.self_attention(x1, x1, x1, mask))
    x3 = self.norm2(x2)
    x4 = src + self.dropout2(self.feed_forward(x3))
    return x4

## Enocder Transformer

class Encoder(nn.Module):
  def __init__(self, max_seq_len, vocab_size, embed_dim, num_layers, num_heads, dropout=0.1):
    """
    Args:
      max_seq_len: Max length of input/output sequences, used for positional encoding.
      vocab_size: total unique token in vocab
      embed_dim: dimensionality of the embedding vector
      num_layers: number of layers in decoder
      num_heads: number of attention heads (8)
    """
    super(Encoder, self).__init__()
    self.embedding = InputEmbeddings(vocab_size, embed_dim)
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
    self.norm = Norm(embed_dim)
    self.position_embedding = PositionalEncoding(max_seq_len, embed_dim, dropout)

  def forward(self, src, src_mask):
    # embed input
    x = self.embedding(src)
    # add Positional encoding
    x = self.position_embedding(x)
    # propagate linear layers
    for layer in self.layers:
      x =  layer(x, src_mask)
    # normalization
    x = self.norm(x)
    return x