import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def loss_fn(logits, target):
    B, S, C = logits.shape
    # TODO: implement cross entropy directly
    return F.cross_entropy(logits.view(B * S, C), target.view(-1))


@dataclass(frozen=True)
class NanoGPTConfig:
    vocab_size: int
    emb_size: int
    block_size: int
    num_layers: int
    num_heads: int
    dropout_ratio: float
    use_flash_attn: bool
    group_size: int = 1

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0")
        if self.emb_size <= 0:
            raise ValueError("emb_size must be greater than 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be greater than 0")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be greater than 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")
        if self.group_size <= 0:
            raise ValueError("group_size must be greater than 0")
        if self.group_size > self.num_heads:
            raise ValueError("group_size cannot exceed num_heads")
        if self.num_heads % self.group_size != 0:
            raise ValueError("num_heads must be divisible by group_size")
        if self.emb_size % self.num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")
        if self.emb_size % self.group_size != 0:
            raise ValueError("emb_size must be divisible by group_size")
        if not (0.0 <= self.dropout_ratio <= 1.0):
            raise ValueError("dropout_ratio must be in [0, 1]")


class MultiHeadAttention(nn.Module):
    # add local parameters: block_size
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.config = config
        self.use_gqa = (config.group_size > 1) and config.use_flash_attn
        self.head_size = config.emb_size // config.num_heads

        if self.use_gqa:
            self.qkv_head = nn.Linear(
                config.emb_size,
                config.emb_size + config.emb_size * 2 // config.group_size,
            )
        else:
            self.qkv_head = nn.Linear(config.emb_size, config.emb_size * 3)

        self.proj = nn.Linear(config.emb_size, config.emb_size)
        # Question: can dropout with same ratio be merged? does it affect gradient when merging
        self.attn_dropout = nn.Dropout(config.dropout_ratio)
        self.proj_dropout = nn.Dropout(config.dropout_ratio)
        if not config.use_flash_attn:
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(config.block_size, config.block_size).view(
                        1, 1, config.block_size, config.block_size
                    )
                ),
            )

    def forward(self, x):
        B, S, C = x.shape
        if self.use_gqa:
            qkv_prod = self.qkv_head(x)  # B, S, Mixed
            q = (
                qkv_prod[:, :, : self.config.emb_size]
                .view(B, S, self.config.num_heads, -1)
                .permute(0, 2, 1, 3)
            )  # B, H, S, C
            kv_width = self.config.emb_size // self.config.group_size
            k = (
                qkv_prod[:, :, self.config.emb_size : self.config.emb_size + kv_width]
                .view(B, S, self.config.num_heads // self.config.group_size, -1)
                .permute(0, 2, 1, 3)
            )  # B, H/G, S, C
            v = (
                qkv_prod[:, :, self.config.emb_size + kv_width :]
                .view(B, S, self.config.num_heads // self.config.group_size, -1)
                .permute(0, 2, 1, 3)
            )  # B, H/G, S, C

        else:
            qkv = (
                self.qkv_head(x)
                .view(B, S, 3, self.config.num_heads, C // self.config.num_heads)
                .permute(2, 0, 3, 1, 4)
            )  # 3, B, N, S, H
            q, k, v = qkv[0], qkv[1], qkv[2]  # B, N, S, H

        if self.config.use_flash_attn:
            dot_product = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout_ratio if self.training else 0,
                is_causal=True,
                enable_gqa=self.use_gqa,
            )
        else:
            # TODO: implement other attention, e.g. MQA, GQA, MLA, delta attention
            qk_product = q @ k.transpose(-2, -1) / math.sqrt(C // self.config.num_heads)
            self.attention = F.softmax(
                qk_product.masked_fill(self.bias[:, :, :S, :S] == 0, float("-inf")),
                dim=-1,
            )  # B, N, S, S
            dot_product = self.attn_dropout(self.attention) @ v  # B, N, S, H

        dot_product = dot_product.transpose(-2, -3).reshape(
            B, S, -1
        )  # B, S, N * H -> B, S, C
        # Question: add a non-linear layer between attn and final projection
        return self.proj_dropout(self.proj(dot_product))


class DecoderBlock(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.config = config

        # TODO: Visualize attention weight
        self.MHA = MultiHeadAttention(config)
        self.proj = nn.Sequential(
            nn.Linear(config.emb_size, 4 * config.emb_size),
            nn.GELU(),
            nn.Linear(4 * config.emb_size, config.emb_size),
            nn.Dropout(config.dropout_ratio),
        )
        # Question: how to compute gradient for layer/batch norm?
        # TODO: implement layer norm directly
        # TODO: compare layer norm with RMS norm, why pick RMS norm?
        self.alpha = nn.Parameter(torch.ones(2, 1))
        self.beta = nn.Parameter(torch.ones(2, 1))

    def forward(self, x):
        # X: B, S, C
        x = self.alpha[0] * x + self.beta[0] * self.MHA(F.rms_norm(x, (x.size(-1),)))
        # return self.alpha[1] * x + self.beta[1] * self.proj(self.ln2(x))
        return x + self.proj(F.rms_norm(x, (x.size(-1),)))


class NanoGPT(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.emb_size)
        # TODO: implement RoPE, sinusoidal
        self.position_embedding_table = nn.Embedding(config.block_size, config.emb_size)
        # TODO: implement hybrid attention with sssl switch
        self.decoder_block = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_layers)
        )
        self.lm_head = nn.Linear(config.emb_size, config.vocab_size)

    def forward(self, input):
        B, S = input.shape
        state = self.token_embedding_table(input)  # B, S, C
        pos_emb = self.position_embedding_table(
            torch.arange(S, device=input.device)
        ).unsqueeze(
            0
        )  # 1, S, E
        state = state + pos_emb
        for i in range(self.config.num_layers):
            state = self.decoder_block[i](state)
        return self.lm_head(
            F.rms_norm(state, (state.size(-1),))
        )  # BATCH x SEQ_LEN x VOCAB_SIZE

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits = self(idx[:, -self.config.block_size :])  # logits: B, S, V
            logits = logits[:, -1, :]  # BATCH x VOCAB_SIZE
            probs = F.softmax(logits, dim=-1)  # BATCH x VOCAB_SIZE
            token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, token), dim=1)  # append to sequence
        return idx
