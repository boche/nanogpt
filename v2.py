import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
emb_size = 512 
batch_size = 32
block_size = 16
num_head = 8
seed = 42
decoder_layer = 6
lr = 3e-4
dropout_ratio = 0.0
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def build_vocab(text):
    chars = sorted(list(set(text)))
    ch2i = {ch: i for i, ch in enumerate(chars)}
    i2ch = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [ch2i[c] for c in s]
    decode = lambda s: "".join(i2ch[i] for i in s)
    return chars, encode, decode

class MultiHeadAttention(torch.nn.Module):
    # add local parameters: block_size
    def __init__(self, num_head, emb_size, head_size):
        super().__init__()
        mask = torch.zeros(block_size, block_size).masked_fill(
            torch.tril(torch.ones(block_size, block_size)) == 0, float('-inf')
        )
        # TODO: merge q, k, v linear
        self.q_head = nn.Linear(emb_size, emb_size)
        self.k_head = nn.Linear(emb_size, emb_size)
        self.v_head = nn.Linear(emb_size, emb_size)
        self.register_buffer('mask', mask)
        self.proj = torch.nn.Linear(num_head * head_size, emb_size)
        # Question: can dropout with same ratio be merged? does it affect gradient when merging
        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.proj_dropout = nn.Dropout(dropout_ratio)
    
    def forward(self, x):
        B, S, C = x.shape
        q = self.q_head(x).view(B, S, num_head, C // num_head).transpose(1, 2) # B, N, S, H
        k = self.k_head(x).view(B, S, num_head, C // num_head).transpose(1, 2) # B, N, S, H
        v = self.v_head(x).view(B, S, num_head, C // num_head).transpose(1, 2) # B, N, S, H
        # TODO: add positional encoding

        # TODO: verify the mask is working as expected
        # TODO: flash attention
        # TODO: Don't store self attention because it's expensive
        self.attention = F.softmax(q @ k.transpose(-2,-1) / math.sqrt(C // num_head) + self.mask, dim=-1) # B, N, S, S
        dot_product = self.attn_dropout(self.attention) @ v # B, N, S, H
        dot_product = dot_product.transpose(-2,-3).reshape(B, S, -1) # B, S, N * H
        # Question: add a non-linear layer between attn and final projection
        return self.proj_dropout(self.proj(dot_product))


class DecoderBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Visualize attention weight
        self.MHA = MultiHeadAttention(num_head, emb_size, emb_size // num_head)
        self.proj = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            # TODO: try GELU
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout_ratio)
        )
        # Question: how to compute gradient for layer/batch norm?
        # TODO: implement layer norm directly
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)
        

    def forward(self, x):
        # X: B, S, C
        x = x + self.MHA(self.ln1(x))
        return x + self.proj(self.ln2(x))
    

class NanoGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        self.decoder_block = nn.ModuleList(
            DecoderBlock() for i in range(decoder_layer)
        )
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, input, target):
        state = self.token_embedding_table(input) # BATCH x SEQ_LEN x EMB_SIZE
        for i in range(decoder_layer):
            state = self.decoder_block[i](state)
        logits = self.lm_head(state) # BATCH x SEQ_LEN x VOCAB_SIZE

        if target is None:
            return logits, None
        else:
            B, S, C = logits.shape
            # TODO: implement cross entropy directly
            loss = F.cross_entropy(logits.view(B*S, C), target.view(-1))
            return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            # TODO: handle case when idx has fewer than block_size input
            logits, _ = self(idx[:, -block_size:], None) # logits: B, S, V
            logits = logits[:, -1, :] # BATCH x VOCAB_SIZE
            probs = F.softmax(logits, dim=-1) # BATCH x VOCAB_SIZE
            token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, token), dim=1) # append to sequence
        return idx

def get_batch(text, split):
    train, val = text[:int(0.9*len(text))], text[int(0.9*len(text)):]
    data = train if split == 'train' else val

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    # TODO: implement this more natively in matrix
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+1+block_size] for i in ix]) # b * seq_len

    return xb, yb

with urllib.request.urlopen(url) as response:
    text = response.read().decode("utf-8")

chars, encode, decode = build_vocab(text)
vocab_size = len(chars)
# TODO: don't store unnecessary data on gpu
text_indices = torch.tensor(encode(text), dtype=torch.long).to(device) 
torch.manual_seed(seed)

model = NanoGPT(vocab_size).to(device)
torch.set_float32_matmul_precision('high')
# model = torch.compile(model)

# TODO: print model parameters size
# TODO: use gradient clipping to stablize model, how to simulate gradient instability
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# TODO: add monitoring for running time
# TODO: add monitoring for maxmimal gpu memory used
# TODO: try low precision representation
# TODO: try torch.compile

torch.cuda.reset_peak_memory_stats(device)
t0 = time.time()

cum_loss = 0
num_eval_per_print = 100
num_train_per_print = 500

for i in range(2000):
    xb, yb = get_batch(text_indices, 'train')
    logits, loss = model(xb, yb)
    cum_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % num_train_per_print == 0:
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            num_eval = 100
            for j in range(num_train_per_print):
                xb, yb = get_batch(text_indices, 'test')
                _, loss = model(xb, yb)
                eval_loss += loss

        print(f"step {i}, train loss {cum_loss / num_train_per_print: .4f}, eval loss {eval_loss/num_train_per_print: .4f}, time: {time.time() - t0: .2f} seconds")
        cum_loss = 0
        model.train()

model.eval()
outputs = model.generate(torch.ones((batch_size, block_size), dtype=torch.long).to(device), max_new_tokens=100)

# for output in outputs:
#     print(decode(output.view(-1).tolist()))

torch.cuda.synchronize(device)
print('-'*80)
print(int(torch.cuda.max_memory_allocated(device)/1024**2), "MiB")
print('Time consumed: %.2f seconds' % (time.time() - t0))