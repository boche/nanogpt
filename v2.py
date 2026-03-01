from dataclasses import dataclass, replace
import urllib.request
import torch
import time
from NanoGPT import NanoGPT, NanoGPTConfig
from helper import loss_fn, grad_norm, weight_norm
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    eval_interval: int = 1000
    train_iter: int = 4000
    eval_iter: int = 100
    batch_size: int = 128
    lr: float = 3e-4
    use_bf16: bool = True
    model_path: str = "model.pt"
    url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    )
    tb_output_path: str = "/mnt/c/Users/aimbo/Code/nanogpt/tb_output/"


model_cfg = NanoGPTConfig(
    vocab_size=1,
    emb_size=256,
    block_size=256,
    num_layers=6,
    num_heads=8,
    dropout_ratio=0.2,
    use_flash_attn=True,
    group_size=2,
)


def build_vocab(text):
    chars = sorted(list(set(text)))
    ch2i = {ch: i for i, ch in enumerate(chars)}
    i2ch = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [ch2i[c] for c in s]
    decode = lambda s: "".join(i2ch[i] for i in s)
    return chars, encode, decode


def get_batch(text, split, batch_size, block_size):
    train, val = text[: int(0.9 * len(text))], text[int(0.9 * len(text)) :]
    data = train if split == "train" else val

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    offset = torch.arange(block_size)
    pos = ix[:, None] + offset[None, :]

    return data[pos], data[pos + 1]


train_cfg = TrainConfig()

# model and runtime config
device = "cuda" if torch.cuda.is_available() else "cpu"

with urllib.request.urlopen(train_cfg.url) as response:
    text = response.read().decode("utf-8")

chars, encode, decode = build_vocab(text)
model_cfg = replace(model_cfg, vocab_size=len(chars))

text_indices = torch.tensor(encode(text), dtype=torch.long)
torch.manual_seed(train_cfg.seed)


model = NanoGPT(model_cfg).to(device)
torch.set_float32_matmul_precision("high")
model = torch.compile(model)

# TODO: try other optimizer
# TODO: understand AdamW better
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=(device == "cuda"),
)

# TODO: try low precision representation
torch.cuda.reset_peak_memory_stats(device)
t0 = time.time()

cum_loss = 0
writer = SummaryWriter(
    log_dir=train_cfg.tb_output_path + f"nanogpt_{time.strftime('%m%d_%H%M')}"
)

for i in range(1, 1 + train_cfg.train_iter):
    xb, yb = get_batch(
        text_indices, "train", train_cfg.batch_size, model_cfg.block_size
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=train_cfg.use_bf16
    ):
        logits = model(xb.to(device))
    loss = loss_fn(logits, yb.to(device))
    cum_loss += loss.item()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if i % train_cfg.eval_interval == 0:
        # todo: attention entropy check, activation check
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            for j in range(train_cfg.eval_iter):
                xb, yb = get_batch(
                    text_indices, "test", train_cfg.batch_size, model_cfg.block_size
                )

                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=train_cfg.use_bf16
                ):
                    logits = model(xb.to(device))

                eval_loss += loss_fn(logits, yb.to(device))

        train_loss = cum_loss / train_cfg.eval_interval
        eval_loss = eval_loss / train_cfg.eval_iter
        print(
            f"step {i}, train loss {train_loss: .4f}, eval loss {eval_loss: .4f}, time: {time.time() - t0: .2f} seconds"
        )
        writer.add_scalar("train/loss", train_loss, i)
        writer.add_scalar("train/grad_norm", grad_norm(model), i)
        writer.add_scalar("train/weight_norm", weight_norm(model), i)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], i)
        writer.add_scalar("train/activation_first", model.activation_first.item(), i)
        writer.add_scalar("train/activation_last", model.activation_last.item(), i)
        writer.add_scalar("eval/loss", eval_loss, i)
        cum_loss = 0

writer.close()
torch.save(model.state_dict(), train_cfg.model_path)
model.load_state_dict(torch.load(train_cfg.model_path, map_location=device))
model.to(device)

generate_start_time = time.time()
model.eval()
with torch.autocast(
    device_type="cuda", dtype=torch.bfloat16, enabled=train_cfg.use_bf16
):
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

# print(decode(outputs[0].tolist()))
