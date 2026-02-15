import urllib.request
import torch
import time
from NanoGPT import NanoGPT, loss_fn


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42

# model param
emb_size = 256
block_size = 256
decoder_layer = 6
dropout_ratio = 0.2
lr = 3e-4

# train param
eval_interval = 1000
train_iter = 4000
eval_iter = 100
batch_size = 128

# atttention param
num_head = 8
use_flash_attn = True
group_size = 2
use_gqa = group_size > 1

use_bf16 = True
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def build_vocab(text):
    chars = sorted(list(set(text)))
    ch2i = {ch: i for i, ch in enumerate(chars)}
    i2ch = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [ch2i[c] for c in s]
    decode = lambda s: "".join(i2ch[i] for i in s)
    return chars, encode, decode


def get_batch(text, split):
    train, val = text[: int(0.9 * len(text))], text[int(0.9 * len(text)) :]
    data = train if split == "train" else val

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    # TODO: implement this more natively in matrix
    xb = torch.stack([data[i : i + block_size] for i in ix])
    yb = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])  # b * seq_len

    return xb, yb


with urllib.request.urlopen(url) as response:
    text = response.read().decode("utf-8")

chars, encode, decode = build_vocab(text)
vocab_size = len(chars)
text_indices = torch.tensor(encode(text), dtype=torch.long)
torch.manual_seed(seed)

model = NanoGPT(
    vocab_size,
    emb_size,
    block_size,
    decoder_layer,
    num_head,
    dropout_ratio,
    use_flash_attn,
    group_size,
).to(device)
torch.set_float32_matmul_precision("high")
model = torch.compile(model)

# TODO: try other optimizer
# TODO: understand AdamW better
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=(device == "cuda"),
)

# TODO: try low precision representation
torch.cuda.reset_peak_memory_stats(device)
t0 = time.time()

cum_loss = 0

for i in range(1, 1 + train_iter):
    xb, yb = get_batch(text_indices, "train")
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(xb.to(device))
    loss = loss_fn(logits, yb.to(device))
    cum_loss += loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if i % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            for j in range(eval_iter):
                xb, yb = get_batch(text_indices, "test")

                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
                ):
                    logits = model(xb.to(device))

                eval_loss += loss_fn(logits, yb.to(device))

        print(
            f"step {i}, train loss {cum_loss/eval_interval: .4f}, eval loss {eval_loss/eval_iter: .4f}, time: {time.time() - t0: .2f} seconds"
        )
        cum_loss = 0
        model.train()

generate_start_time = time.time()
model.eval()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
    outputs = model.generate(
        torch.ones((8, 1), dtype=torch.long).to(device), max_new_tokens=10000
    )
generate_end_time = time.time()

torch.cuda.synchronize(device)
print("-" * 80)
param_count = sum(p.numel() for p in model.parameters())
print(f"parameters: {param_count:,}")
print(int(torch.cuda.max_memory_allocated(device) / 1024**2), "MiB")
print(
    "Train time: %.2f seconds, Generate time: %.2f seconds"
    % (generate_start_time - t0, generate_end_time - generate_start_time)
)
print(
    "Train time: %.2f seconds, Generate time: %.2f seconds"
    % (generate_start_time - t0, generate_end_time - generate_start_time)
)

# for output in outputs:
#     print(decode(output.view(1).tolist()))
