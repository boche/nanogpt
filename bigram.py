import urllib.request
import torch
import torch.nn.functional as F


# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
emb_size = 128
batch_size = 16
block_size = 8
seed = 42
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def build_vocab(text):
    chars = sorted(list(set(text)))
    ch2i = {ch: i for i, ch in enumerate(chars)}
    i2ch = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [ch2i[c] for c in s]
    decode = lambda s: "".join(i2ch[i] for i in s)
    return chars, encode, decode

class BigramModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, emb_size)
        self.lm_head = torch.nn.Linear(emb_size, vocab_size)

    def forward(self, input, target):
        tok_emb = self.token_embedding_table(input) # BATCH x SEQ_LEN x EMB_SIZE
        logits = self.lm_head(tok_emb) # BATCH x SEQ_LEN x VOCAB_SIZE

        if target is None:
            return logits, None
        else:
            B, S, C = logits.shape
            loss = F.cross_entropy(logits.view(B*S, C), target.view(-1))
            return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, _ = self(idx, None)
            logits = logits[:, -1, :] # BATCH x VOCAB_SIZE
            probs = F.softmax(logits, dim=-1) # BATCH x VOCAB_SIZE
            token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, token), dim=1) # append to sequence
        return idx

def get_batch(text, split):
    train, val = text[:int(0.9*len(text))], text[int(0.9*len(text)):]
    data = train if split == 'train' else val

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    xb = torch.stack([data[i:i+block_size] for i in ix])
    yb = torch.stack([data[i+1:i+1+block_size] for i in ix]) # b * seq_len

    return xb, yb

with urllib.request.urlopen(url) as response:
    text = response.read().decode("utf-8")

chars, encode, decode = build_vocab(text)
vocab_size = len(chars)
text_indices = torch.tensor(encode(text), dtype=torch.long).to(device) 
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model=BigramModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

cum_loss = 0
for i in range(10000):
    xb, yb = get_batch(text_indices, 'train')
    logits, loss = model(xb, yb)
    cum_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"step {i}, loss {cum_loss / 100: .4f}")
        cum_loss = 0

outputs = model.generate(torch.zeros((5, 1), dtype=torch.long).to(device), max_new_tokens=10)

for output in outputs:
    print(decode(output.view(-1).tolist()))