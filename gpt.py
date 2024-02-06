import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""hyperparameter"""
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions? (T)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

# 1 batch = 1 block size of n

# B: Batch
# T: Time: block_size (# tokens in 1 input)
# C: Channel, embedding dimension

class Head(nn.Module):
    """single head masked self-attention"""

    def __init__(self, dropout, n_embd, head_size, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)

        self.dropout = nn.Dropout(dropout)

        # register one instance with 'tril' name, this matrix is a unit lower tringle matrix
        self.register_buffer('trill', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        
        B, T, C = x.shape

        key = self.key(x) # (B, T, hs)
        query = self.query(x) # (B, T, hs)
        value = self.value(x) # (B, T, hs)

        # attention matrix
        wei = query @ key.transpose(-2,-1)  # (B, T, hs) @ (B, hs, T) ---> (B, T, T)

        # masked attention matrix
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=1) # (B, T, T)
        wei = self.dropout(wei) # to reduce some communication between token -> reduce overfit
        out = wei @ value # (B, T, T) @ (B, T, hs) ---> (B, T, hs)

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(dropout, n_embd, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd) # match dimension for residual connection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, T, C/n) ---> (concate n heads in last dimension): (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(self.dropout(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        
        # token level, each token pass individually
        # intuition: after passing through MultiHeadAttention
        # each child (results of each head) from same token hasn't communicated to each other
        # now pass that concatenated result to a linear layer for it to learn and talk to each other

        # (B, T, C) --- (B, T, C)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # (B, T, C)
        return self.net(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
         super().__init__()

         head_size = n_embd // n_head
         self.sa = MultiHeadAttention(n_embd, n_head, head_size, dropout, block_size)
         self.ffwd = FeedForward(n_embd, dropout)

         # apply across the embedding dimension for each individual token
         self.ln1 = nn.LayerNorm(n_embd)
         self.ln2 = nn.LayerNorm(n_embd)
         
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size,
                block_size=256,
                n_embd=384,
                n_head=6,
                n_layer=6,
                dropout=0.2
                ):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[TransformerDecoderBlock(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.shape
        
        # idx, taget: (B, T)
        
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(idx) # (T,C)

        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if target == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx (B, T): indices of current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self.forward(idx)
            # get last prediction only
            logits = logits[:, -1, :] # (B, 1, C) --- (B, C)
            probs = F.softmax(logits) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append to the current sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    
