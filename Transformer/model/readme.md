# Transformer - Pytorch Implementation
This is a PyTorch implementation of the Transformer model in the paper Attention is All You Need (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017).


<p align="center">
<img src="https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png" width="700">
</p>


The directory structure of this project is shown below:
```bash
📂 transformer
│── 📂 model
│   │── __init__.py
│   │── embedding_layer.py     <-- Input Embedding + Positional Encoding + Norm
│   │── self_attention.py      <-- Multi-Head Attention
│   │── encoder.py             <-- Encoder Layer + Full Encoder
│   │── decoder.py             <-- Decoder Layer + Full Decoder
│   │── transformer.py         <-- Full Transformer Model
```

---
# Models

# Input Embedding
First of all, we need to convert each word in input sequence to an embedding vector. The embedding vector create a semantic representation of words.

Suppoese each embedding vector is of 512 dimension and suppose our vocab size is 11, then our embedding matrix will be of size 11x512. These marix will be learned on training and during inference each word will be mapped to corresponding 512 d vector. If batch_size = 32 and vocab_size= 11, the output shape becomes (32, 11, 512)


> The paper (Attention Is All You Need) scales embeddings by multiplying them with √d_model (where d_model = embedding_dim).
$$
E(x) = Embedding(x) * {\sqrt{d_{model}}}
$$
```python
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
```
## Positional Encoding
Transformers do not have recurrence (RNNs) or CNNs, so they do not inherently understand word order. To solve this, positional encoding (PE) is added to word embeddings which allow the model to know the position of words in a sequence.

Inorder for the model to make sense of the sentence, it needs to know two things about the each word.

- what does the word mean?
- what is the position of the word in the sentence.
In "attention is all you need paper" author used the following functions to create positional encoding. On odd time steps a cosine function is used and in even time steps a sine function is used.
> As mentioned in the paper `The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.`


<!-- $$PE_{(pos, 2i)}=sin(\frac{pos}{10000^{2i/d_{model}}})$$
$$PE_{(pos, 2i+1)}=cos(\frac{pos}{10000^{2i/d_{model}}})$$ -->

<p align="center">
<img src="https://latex.codecogs.com/png.image?\large&space;\dpi{110}\bg{white}\begin{}\\PE_{(pos,&space;2i)}=sin(\frac{pos}{10000^{2i/d_{model}}})&space;\\PE_{(pos,&space;2i&plus;1)}=cos(\frac{pos}{10000^{2i/d_{model}}})\end{}" title="https://latex.codecogs.com/png.image?\large \dpi{110}\bg{white}\begin{}\\PE_{(pos, 2i)}=sin(\frac{pos}{10000^{2i/d_{model}}}) \\PE_{(pos, 2i+1)}=cos(\frac{pos}{10000^{2i/d_{model}}})\end{}" />
</p>
- pos → Position of the word in the sequence <br>
- i → The dimension index (divided into even and odd parts)<br>
- d_model → Embedding size (i.e 512)<br>
- 10000 → A large constant that scales down positions smoothly<br>


```python
# The positional encoding vector, embedding_dim is d_model
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
```

## ** Self Attention (Scaled Product Attention)**
By applying self-attention, the transformer model can capture the dependencies between different words in the input sequence and learn to focus on the most relevant words for each position. This helps in understanding the context and improving the quality of translation or any other sequence-based task.


The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. Each of the vector will be of dimension 1x64. `(dim/8) = 64`

For each token in the input sequence, self-attention computes:
- Query (Q) → "What am I looking for?"
- Key (K) → "What information do I have?"
- Value (V) → "What is my actual content?
- 
**The Attention Score is computed as:**
<!-- $$Attention(Q, K, V ) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$ -->
<p align="center">
<img src="https://latex.codecogs.com/png.image?\large&space;\dpi{110}\bg{white}Attention(Q,&space;K,&space;V&space;)&space;=&space;softmax(\frac{QK^T}{\sqrt{d_k}})V" title="https://latex.codecogs.com/png.image?\large \dpi{110}\bg{white}Attention(Q, K, V ) = softmax(\frac{QK^T}{\sqrt{d_k}})V" />
</p>


## **Multi-Head Attention**
SInce of applying self-attention once, Multi-Head Attention (MHA) applies it multiple times in parallel, each with different learned projections.
1. MHA allows the model to focus on different parts of the sentence in different ways.
2. It Helps capture various linguistic relationships, like subject-verb agreement, synonyms, etc.
<!-- $$MultiHead(Q, K, V ) = Concat(head_1,..., head_h)W_O$$

$$head_i = Attention(QWQ_i^Q, KW^K_i,VW^V_i)$$ -->

<p align="center">
<img src="https://latex.codecogs.com/png.image?\large&space;\dpi{110}\bg{white}\begin{}\\MultiHead(Q,&space;K,&space;V&space;)&space;=&space;Concat(head_1,...,&space;head_h)W_O&space;\\head_i&space;=&space;Attention(QWQ_i^Q,&space;KW^K_i,VW^V_i)\end{}" title="https://latex.codecogs.com/png.image?\large \dpi{110}\bg{white}\begin{}\\MultiHead(Q, K, V ) = Concat(head_1,..., head_h)W_O \\head_i = Attention(QWQ_i^Q, KW^K_i,VW^V_i)\end{}" />
</p>

```python
# Multi-head attention layer
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
```
# **Encoder**
## **Encoder Layer**
Each Encoder Layer consists of two main sub-layers:

- Multi-Head Self-Attention: Helps each word attend to all others.

- Feed-Forward Network (FFN): Applies two linear transformations with a ReLU activation in between.

🔹 How It Works
1. Input goes through LayerNorm.

2. Self-attention mechanism is applied.

3. Another LayerNorm, then the output passes through the FFN.
   
![alt text](image.png)
```python
# Transformer encoder layer
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
```

<!-- <p align="center">
<img src="https://www.factored.ai/wp-content/uploads/2021/09/image2-580x1024.png" width="350">
</p> -->



### Encoder
<figure>
<p align="center">
<img src="https://kikaben.com/transformers-encoder-decoder/images/encoder-layer-norm.png" width="350">
</p>
<figcaption>
Encoder: The encoder is composed of a stack of <b>N = 6</b> identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension.
</figcaption>
</figure>
```python
# Encoder transformer
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, num_heads, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList([EncoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
    
    def forward(self, source, source_mask):
        # Embed the source
        x = self.embedding(source)
        # Add the position embeddings
        x = self.position_embedding(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, source_mask)
        # Normalize
        x = self.norm(x)
        return x
```
# **Decoder**
## Decoder Layer
Decoder: The decoder is also composed of a stack of *N = 6* identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position *i* can depend only on the known outputs at positions less than *i*.
![alt text](image-1.png)
```python
# Transformer decoder layer
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
```
### Decoder
```python
# Decoder transformer
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len,num_heads, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, num_heads, 2048, dropout) for _ in range(num_layers)])
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)
    def forward(self, target, memory, source_mask, target_mask):
        # Embed the source
        x = self.embedding(target)
        # Add the position embeddings
        x = self.position_embedding(x)
        # Propagate through the layers
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        # Normalize
        x = self.norm(x)
        return x
```

## Transformer

Finally we will arrange all submodules and creates the entire tranformer architecture.


```python
# Transformers
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_max_seq_len, target_max_seq_len, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_vocab_size, embedding_dim, source_max_seq_len, num_heads, num_layers, dropout)
        self.decoder = Decoder(target_vocab_size, embedding_dim, target_max_seq_len, num_heads, num_layers, dropout)
        self.final_linear = nn.Linear(embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src_seq):
        src_mask = (src_seq != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt_seq):
        N, tgt_len = tgt_seq.shape
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)
                              ).expand(N, 1, tgt_len, tgt_len)
        return tgt_mask.to(self.device)
    def forward(self, source, target, source_mask, target_mask):
        # Encoder forward pass
        memory = self.encoder(source, source_mask)
        # Decoder forward pass
        output = self.decoder(target, memory, source_mask, target_mask)
        # Final linear layer
        output = self.dropout(output)
        output = self.final_linear(output)
        return output
```

