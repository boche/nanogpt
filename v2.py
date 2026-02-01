import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
emb_size = 512 
batch_size = 64
block_size = 128
num_head = 8
decoder_layer = 6
lr = 3e-4
dropout_ratio = 0.2
eval_interval = 500
train_iter = 5000
eval_iter = 100
use_flash_attn = True
use_bf16 = True
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
        self.qkv_head = nn.Linear(emb_size, emb_size * 3)
        self.proj = torch.nn.Linear(num_head * head_size, emb_size)
        # Question: can dropout with same ratio be merged? does it affect gradient when merging
        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.proj_dropout = nn.Dropout(dropout_ratio)
        if not use_flash_attn:
            self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)))
    
    def forward(self, x):
        B, S, C = x.shape
        qkv = self.qkv_head(x).view(B, S, 3, num_head, C // num_head).permute(2, 0, 3, 1, 4) # 3, B, N, S, H
        q, k, v = qkv[1], qkv[1], qkv[2]

        if use_flash_attn:
            # TODO: implement flash attention with Triton
            dot_product = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_ratio if self.training else 0, is_causal=True)
        else:
            # TODO: Don't store self attention because it's expensive
            # TODO: verify the mask is working as expected
            # TODO: implement other attention, e.g. MQA, GQA, MLA, delta attention
            self.attention = F.softmax(q @ k.transpose(-2,-1) / math.sqrt(C // num_head), dim=-1).masked_fill(
                self.bias[:, :, :B, :B] == 0, float('-inf')
            ) # B, N, S, S
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
        self.alpha = nn.Parameter(torch.ones(2,1))
        self.beta = nn.Parameter(torch.ones(2,1))

    def forward(self, x):
        # X: B, S, C
        x = self.alpha[0] * x + self.beta[0] * self.MHA(self.ln1(x))
        # return self.alpha[1] * x + self.beta[1] * self.proj(self.ln2(x))
        return x + self.proj(self.ln2(x))
    
class NanoGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        # TODO: implement RoPE, sinusoidal
        self.position_embedding_table = nn.Embedding(block_size, emb_size)
        self.decoder_block = nn.ModuleList(
            DecoderBlock() for i in range(decoder_layer)
        )
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, input):
        B, S = input.shape
        state = self.token_embedding_table(input) # BATCH x SEQ_LEN x EMB_SIZE
        pos_emb = self.position_embedding_table(torch.arange(S, device=device)).unsqueeze(0) # 1, S, E
        for i in range(decoder_layer):
            state = self.decoder_block[i](state+pos_emb)
        return self.lm_head(state) # BATCH x SEQ_LEN x VOCAB_SIZE

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits = self(idx[:, -block_size:]) # logits: B, S, V
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

def loss_fn(logits, target):
    B, S, C = logits.shape
    # TODO: implement cross entropy directly
    return F.cross_entropy(logits.view(B*S, C), target.view(-1))

with urllib.request.urlopen(url) as response:
    text = response.read().decode("utf-8")

chars, encode, decode = build_vocab(text)
vocab_size = len(chars)
# TODO: don't store unnecessary data on gpu
text_indices = torch.tensor(encode(text), dtype=torch.long)
torch.manual_seed(seed)

model = NanoGPT(vocab_size).to(device)
torch.set_float32_matmul_precision('high')
model = torch.compile(model)

# TODO: use gradient clipping to stablize model, how to simulate gradient instability
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# TODO: try low precision representation
torch.cuda.reset_peak_memory_stats(device)
t0 = time.time()

cum_loss = 0

for i in range(train_iter):
    xb, yb = get_batch(text_indices, 'train')
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(xb.to(device))
    loss = loss_fn(logits, yb.to(device))
    cum_loss += loss.item()

    loss.backward()
    optimizer.step()

    if i % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            for j in range(eval_iter):
                xb, yb = get_batch(text_indices, 'test')

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(xb.to(device))

                eval_loss += loss_fn(logits, yb.to(device))

        print(f"step {i}, train loss {cum_loss/eval_interval: .4f}, eval loss {eval_loss/eval_iter: .4f}, time: {time.time() - t0: .2f} seconds")
        cum_loss = 0
        model.train()

generate_start_time = time.time()
model.eval()
with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
    outputs = model.generate(torch.ones((1, 1), dtype=torch.long).to(device), max_new_tokens=10000)
generate_end_time = time.time()

torch.cuda.synchronize(device)
print('-'*80)
param_count = sum(p.numel() for p in model.parameters())
print(f'parameters: {param_count:,}')
print(int(torch.cuda.max_memory_allocated(device)/1024**2), "MiB")
print('Train time: %.2f seconds, Generate time: %.2f seconds' % (generate_start_time - t0, generate_end_time - generate_start_time))

for output in outputs:
    print(decode(output.view(-1).tolist()))