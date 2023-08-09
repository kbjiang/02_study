import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 256 # what is the maximum context length for predictions?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.1

from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(f"tokenizer-en.json")
vocab_size = tokenizer.get_vocab_size()
vocab_size_enc = tokenizer.get_vocab_size()

PAD_IDX = tokenizer.token_to_id("<pad>")
BOS_IDX = tokenizer.token_to_id("<bos>")
EOS_IDX = tokenizer.token_to_id("<eos>")

class Head(nn.Module):
    """ one head of self/cross attention with optional causal masking """

    def __init__(self, head_size, is_causal):
        super().__init__()
        self.is_causal = is_causal
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        # x is the input for query, y is for key and value; x and y can be the same
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        q = self.query(x) # (B,T,hs)
        k = self.key(y)   # (B,T_y,hs)
        v = self.value(y) # (B,T_y,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T_y) -> (B, T, T_y)
        # add causal mask for decoders
        if self.is_causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T_y)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, T_y) @ (B, T_y, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, is_causal=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, is_causal) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        out = torch.cat([h(x, y) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    """ Encoding block """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, is_causal=False)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x), self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class DecoderBlock(nn.Module):
    """ Decoding blcok"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.msa = MultiHeadAttention(n_head, head_size, is_causal=True)
        self.xa = MultiHeadAttention(n_head, head_size, is_causal=False)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)

    def forward(self, x, y):
        # x is decoder input, y is encoder output
        x = x + self.msa(self.ln1(x), self.ln1(x))
        x = x + self.xa(self.ln2(x), self.ln3(y))
        x = x + self.ffwd(self.ln4(x))
        return x

class PositionalEncoding(nn.Module):
    """ compute sinusoid encoding.  """
    def __init__(self, n_embd, device):
        super().__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(block_size, n_embd, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, block_size)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, n_embd, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / n_embd)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / n_embd)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        B, T = x.size()
        return self.encoding[:T, :]

class Transformer(nn.Module):
    """ Transformer with encoder-decoder structure"""

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.token_embedding_table_enc = nn.Embedding(vocab_size_enc, n_embd)
        self.position_embedding = PositionalEncoding(n_embd, device=device)
        self.encoder_blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        # cannot use nn.Sequential for DecoderBlock because it takes two inputs
        self.decoder_blocks = nn.ModuleList([DecoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, idx_enc, targets=None):
        B, T = idx.shape

        tok_emb_enc = self.token_embedding_table_enc(idx_enc) # (B,T_y,C)
        pos_emb_enc = self.position_embedding(idx_enc) # (T_y, c)
        y = tok_emb_enc + pos_emb_enc # (B,T_y,C)
        y = self.encoder_blocks(y) # (B,T_y,C)

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding(idx) # (T, c)
        x = tok_emb + pos_emb # (B,T,C)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, y) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=PAD_IDX)

        return logits, loss

    def generate(self, idx_enc, greedy=False):
        # every generation starts with the BOS token
        B = idx_enc.shape[0]
        idx = torch.ones(B,1).fill_(BOS_IDX).type(torch.long).to(device)
        for i in range(block_size):
            # get the predictions
            logits, _ = self(idx, idx_enc)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            if greedy:
                idx_next = torch.argmax(logits, dim=-1)
            else:
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # once predicts EOS, everything follows becomes EOS
            idx_next = torch.where(idx[:, -1]==EOS_IDX, EOS_IDX, idx_next.squeeze())
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next[:, None]), dim=1) # (B, T+1)
            # stop generation if everything is EOS
            if torch.all(idx[:, -1]==EOS_IDX):
                break
        return idx
