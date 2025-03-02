import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of the embedding vector (e.g., 512 from the paper).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: Input tensor
        Returns:
            out: Scaled embedding vector.
        """

        # Multiply the embedding by sqrt(embedding_dim) as described in the paper.
        return self.embedding(x) * math.sqrt(self.embedding_dim)


# register buffer in Pytorch ->
# If you have parameters in your model, which should be saved and restored in the state_dict,
# but not trained by the optimizer, you should register them as buffers.

class PositionalEncoding(nn.Module):
  def __init__(self, max_seq_len, embed_dim, dropout):
    """
    Args:
      max_seq_len: length of input sequence
      embed_dim: demension of embedding
      dropout: prevents overfitting
    """
    super(PositionalEncoding, self).__init__()
    self.max_seq_len = max_seq_len
    self.embed_dim = embed_dim
    self.dropout = nn.Dropout(dropout)

    pe = torch.zeros(max_seq_len, embed_dim)  # matrix of zeros(rows, columns) rows = seq_len and columns= 512
    # create position of sequence
    for pos in range(max_seq_len):
      for i in range(0, embed_dim, 2):
        pe[pos, i] = math.sin(pos/ (10000 ** ((2 * i)/ self.embed_dim)))
        pe[pos, i + 1] = math.cos(pos/ (10000 ** ((2 * (i + 1))/ self.embed_dim)))
    pe.unsqueeze(0)
    self.register_buffer('pe', pe)  # Register buffer so that it's not a trainable parameter

  def forward(self, x):
    """
    Args:
      x: input vector
    Returns:
      x: output
    """
    # Add the positional encoding vector to the embedding vector
    #  This is a fixed function, not learned weights.
    x = x + (self.pe[:, :x.shape[1], :]).required_grad_(False)
    return self.dropout(x)
  
## Layer Normalization class
class Norm(nn.Module):
  def __init__(self, embed_dim):
    """
    Args:
      embed_dim:
    """
    super(Norm, self).__init__()
    self.norm = nn.LayerNorm(embed_dim)

  def forward(self, x):
    return self.norm(x)
