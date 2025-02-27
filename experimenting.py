from tqdm import tqdm
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.adam

# hyperparameters
batch_size = 256
block_size = 512 # context length
max_iters = 5000
eval_interval = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0.2

# B * T * C * L * 2 * 4 / (1024 ** 3)
param_size = batch_size * block_size * n_embd * n_layer * n_head * 2 * 4 / (1024 ** 3) # Not accurate
print(f"Model parameter size: {param_size:.2f} GB")

# --------------------------------

torch.manual_seed(1337)

with open('amazing.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# with open('chunk10000_2010000.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# Load existance vocab
# stoi, itos = json.load(open(f"{"AR_model"}.json", 'r', encoding='utf-8')).values()
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[str(c)] for c in l])
# vocab_size = len(stoi)
# print(f"Vocab loaded from {"AR_model"}.json")

# here are all the unique characters that occur in this text 
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(stoi.keys())}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])
print(f"data: {len(text)}")
print(f"vocab: {vocab_size}")

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(tqdm = None):
    
    if tqdm is not None:
        tqdm.write('Estimating loss')
    # print('Estimating loss\r', end="")
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss

            if tqdm is not None:
                tqdm.write(f"{split} Estimation: {(k/eval_iters * 100):.2f}% done\r")
            # print(f"{split} Estimation: {(k/eval_iters * 100):.2f}% done\r", end="")

        out[split] = losses.mean()
    model.train()
    # print('\r', end="")
    return out

# Head Attention Layer
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # (B, T, C)
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T) FAccccckkkk why dividing when you have put negative power ???????????
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B, T, H)
        out = wei @ v # (B, T, T) @ (B, T, H) = (B, T, H)
        return out
# MultiHeadAttention Layer
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# FeedForward Layer
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection also
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Block
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else :
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predections
            logits, loss = self(idx_cond)
            # foucs only on the last time step
            logits = logits[:, -1, :]
            # apply softmax
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# -------------- Training --------------
model = BigramLanguageModel()
m = model.to(device)

# Restore model state
# model_save_path = 'AR_model'

# model = BigramLanguageModel()
# m = model.to(device)
# model.load_state_dict(torch.load(f"{model_save_path}.pth", weights_only=True))
# model.eval()
# print(f"Model parameters loaded from {model_save_path}")


num_parameters = sum(p.nelement() for p in model.parameters())
print(f"Parameters: {num_parameters}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# losses = estimate_loss()
# print(f"step {0}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

with tqdm(total=max_iters, desc="Training", unit='iter') as pbar:
    for iter in range(max_iters):

        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            tqdm.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch('train')

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f}")
        pbar.update(1)

# Save model parameters & vocab
model_save_path = 'AR_model_12l_12h'
torch.save(model.state_dict(), f"{model_save_path}.pth")
# with open(f"{model_save_path}.json", 'w', encoding='utf-8') as f:
#     json.dump({'stoi': stoi, 'itos': itos}, f, ensure_ascii=False, indent=4)
print(f"Model parameters saved to {model_save_path}.pth")
# print(f"Vocab saved to {model_save_path}.json")

# Load model
# stoi, itos = json.load(open(f"{model_save_path}.json", 'r', encoding='utf-8')).values()
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[str(c)] for c in l])
# vocab_size = len(stoi)
# print(f"Vocab loaded from {model_save_path}.json")

# model = BigramLanguageModel()
# m = model.to(device)
# model.load_state_dict(torch.load(f"{model_save_path}+1.pth", weights_only=True))
# model.eval()
# print(f"Model parameters loaded from {model_save_path}")

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(encode("السلام"))
# print(decode(m.generate(torch.tensor([encode("السلام")], device=device), max_new_tokens=100)[0].tolist()))

# -------------- prompting --------------

while True:
    prompt = input("Type the start of something or (E) to exit:\n")

    if prompt.lower() == 'e':
        exit()
    else:
        prompt_encoded = torch.tensor([encode(prompt)], device=device)
        answer = decode(m.generate(prompt_encoded, max_new_tokens=100)[0].tolist())
        f = open("outputs.txt", "a", encoding="utf-8")
        f.write(f"{prompt}:\n {answer}\n")
        f.close()
