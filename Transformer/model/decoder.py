import torch
import torch.nn as nn
from model.encoder import EncoderLayer
from model.decoder import DecoderLayer
from model.embedding import Norm, InputEmbeddings, PositionalEncoding
from model.self_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
  def __init__(self, embed_dim, num_heads, dff=2048, dropout=0.1):
    """
    Args:
      embed_dim: dimensionality of the embedding vector
      num_heads: number of attention heads (8)
      dff: dimensionality of feed-forward network (2048)
      dropout: prevent overfitting
    """
    super(DecoderLayer, self).__init__()
    self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
    # Encoder-Decoder Attention
    self.cross_attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
    self.feed_forward = nn.Sequential(
        nn.Linear(embed_dim, dff), # 512, 2048
        nn.ReLU(),
        nn.Linear(dff, embed_dim)
    )
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    self.norm1 = Norm(embed_dim)
    self.norm2 = Norm(embed_dim)
    self.norm3 = Norm(embed_dim)

  def forward(self, x, encoder_output, src_mask, target_mask):
    """
    Args:
      x: input sequence
      encoder_output: output of Encoder
      src_mask: mask for encoder input to ignore padding tokens during self-attention
      target_mask: future tokens masking
    """
    x = self.norm1(x)
    # Add and multihead attention
    x = x + self.dropout1(self.self_attention(x, x, x, target_mask))
    x = self.norm2(x)
    x = x + self.dropout2(self.cross_attn(x, encoder_output, encoder_output, src_mask))
    x = self.norm3(x)
    x= self.dropout3(self.feed_forward(x))
    return x


## Decoder Transformer
class Decoder(nn.Module):
  def __init__(self, max_seq_len, vocab_size, embed_dim, num_layers, num_heads, dropout=0.1):
    """
    Args:
      max_seq_len: Max length of input/output sequences, used for positional encoding.
      vocab_size: total unique token in vocab
      embed_dim: dimensionality of the embedding vector
      num_layers: number of layers in decoder
      num_heads: number of attention heads (8)
    """
    super(Decoder, self).__init__()
    self.embedding = InputEmbeddings(vocab_size, embed_dim)
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
    self.norm = Norm(embed_dim)
    self.position_embedding = PositionalEncoding(max_seq_len, embed_dim, dropout)

  def forward(self, target, encoder_output, src_mask, target_mask):
    # embed input
    x = self.embedding(target)
    # add Positional encoding
    x = self.position_embedding(x)
    # propagate linear layers
    for layer in self.layers:
      x =  layer(x, encoder_output, src_mask, target_mask)
    # normalization
    x = self.norm(x)
    return x