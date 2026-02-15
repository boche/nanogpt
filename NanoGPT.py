import math
import torch
from torch import nn
import torch.nn.functional as F

def loss_fn(logits, target):
    B, S, C = logits.shape
    # TODO: implement cross entropy directly
    return F.cross_entropy(logits.view(B*S, C), target.view(-1))

class MultiHeadAttention(nn.Module):
    # add local parameters: block_size
    def __init__(self, block_size, num_head, emb_size, dropout_ratio, use_flash_attn, group_size):
        super().__init__()

        self.block_size = block_size
        self.num_head = num_head
        self.emb_size = emb_size
        self.dropout_ratio = dropout_ratio
        self.use_flash_attn = use_flash_attn
        self.group_size = group_size
        self.use_gqa = (group_size > 1) & use_flash_attn
        self.head_size = emb_size // num_head

        if self.use_gqa:
            self.qkv_head = nn.Linear(emb_size, emb_size + emb_size * 2 // group_size)
        else:
            self.qkv_head = nn.Linear(emb_size, emb_size * 3)

        self.proj = nn.Linear(emb_size, emb_size)
        # Question: can dropout with same ratio be merged? does it affect gradient when merging
        self.attn_dropout = nn.Dropout(dropout_ratio)
        self.proj_dropout = nn.Dropout(dropout_ratio)
        if not use_flash_attn:
            self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size).view(1, 1, block_size, block_size)))
    
    def forward(self, x):
        B, S, C = x.shape
        if self.use_gqa:
            qkv_prod = self.qkv_head(x) # B, S, Mixed
            q = qkv_prod[:, :, :self.emb_size].view(B, S, self.num_head, -1).permute(0, 2, 1, 3) # B, H, S, C
            k = qkv_prod[:, :, self.emb_size:self.emb_size+self.emb_size//self.group_size].view(B, S, self.num_head//self.group_size, -1).permute(0, 2, 1, 3) # B, H/G, S, C
            v = qkv_prod[:, :, self.emb_size+self.emb_size//self.group_size:].view(B, S, self.num_head//self.group_size, -1).permute(0, 2, 1, 3) # B, H/G, S, C

        else:
            qkv = self.qkv_head(x).view(B, S, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4) # 3, B, N, S, H
            q, k, v = qkv[0], qkv[1], qkv[2] # B, N, S, H

        if self.use_flash_attn:
            # TODO: implement flash attention with Triton
            if self.use_gqa:
                dot_product = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_ratio if self.training else 0, is_causal=True, enable_gqa=self.use_gqa)
            else:
                dot_product = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_ratio if self.training else 0, is_causal=True)

        else:
            # TODO: implement other attention, e.g. MQA, GQA, MLA, delta attention
            qk_product = q @ k.transpose(-2,-1) / math.sqrt(C // self.num_head)
            self.attention = F.softmax(
                qk_product.masked_fill(self.bias[:, :, :S, :S] == 0, float('-inf')),
                dim = -1
            ) # B, N, S, S
            dot_product = self.attn_dropout(self.attention) @ v # B, N, S, H

        dot_product = dot_product.transpose(-2,-3).reshape(B, S, -1) # B, S, N * H -> B, S, C
        # Question: add a non-linear layer between attn and final projection
        return self.proj_dropout(self.proj(dot_product))

class DecoderBlock(nn.Module):
    def __init__(self, block_size, num_head, emb_size, dropout_ratio, use_flash_attn, group_size):
        super().__init__()

        self.block_size = block_size
        self.num_head = num_head
        self.emb_size = emb_size
        self.dropout_ratio = dropout_ratio
        self.use_flash_attn = use_flash_attn
        self.group_size = group_size

        # TODO: Visualize attention weight
        self.MHA = MultiHeadAttention(block_size, num_head, emb_size, dropout_ratio, use_flash_attn, group_size)
        self.proj = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout_ratio)
        )
        # Question: how to compute gradient for layer/batch norm?
        # TODO: implement layer norm directly
        # TODO: compare layer norm with RMS norm, why pick RMS norm?
        self.alpha = nn.Parameter(torch.ones(2,1))
        self.beta = nn.Parameter(torch.ones(2,1))

    def forward(self, x):
        # X: B, S, C
        x = self.alpha[0] * x + self.beta[0] * self.MHA(F.rms_norm(x, (x.size(-1),)))
        # return self.alpha[1] * x + self.beta[1] * self.proj(self.ln2(x))
        return x + self.proj(F.rms_norm(x, (x.size(-1),)))
    
class NanoGPT(nn.Module):
    def __init__(self, vocab_size, emb_size, block_size, decoder_layer, num_head, dropout_ratio, use_flash_attn, group_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.block_size = block_size
        self.decoder_layer = decoder_layer
        self.num_head = num_head
        self.dropout_ratio = dropout_ratio
        self.use_flash_attn = use_flash_attn
        self.group_size = group_size

        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        # TODO: implement RoPE, sinusoidal
        self.position_embedding_table = nn.Embedding(block_size, emb_size)
        # TODO: implement hybrid attention with sssl switch 
        self.decoder_block = nn.ModuleList(
            DecoderBlock(block_size, num_head, emb_size, dropout_ratio, use_flash_attn, group_size) for i in range(decoder_layer)
        )
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, input):
        B, S = input.shape
        state = self.token_embedding_table(input) # B, S, C
        pos_emb = self.position_embedding_table(torch.arange(S, device=input.device)).unsqueeze(0) # 1, S, E
        state = state + pos_emb
        for i in range(self.decoder_layer):
            state = self.decoder_block[i](state)
        return self.lm_head(F.rms_norm(state, (state.size(-1),))) # BATCH x SEQ_LEN x VOCAB_SIZE

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:]) # logits: B, S, V
            logits = logits[:, -1, :] # BATCH x VOCAB_SIZE
            probs = F.softmax(logits, dim=-1) # BATCH x VOCAB_SIZE
            token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, token), dim=1) # append to sequence
        return idx