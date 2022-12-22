from torch import nn
from torch.nn import functional as F
import torch
from torchtyping import TensorType as TT
import math
from einops import rearrange
from fancy_einsum import einsum

VisionInputType = TT["batch", "frame", "image_pos", "d_model"]
LanguageInputType = TT["batch", "pos", "d_model"]
AttentionType = TT["batch", "dest", "src"]


class GatedCrossAttentionDenseLayers(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, d_model: int, expanison_factor: int = 4):
        super().__init__()

        # Alpha parameters
        self.alpha_xattn = nn.Parameter(torch.Tensor([0.]))
        self.alpha_dense = nn.Parameter(torch.Tensor([0.]))

        # Attention layer
        d_head: int = int(d_model / num_heads)
        self.query_layer = nn.Linear(d_model, d_model)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

        # MLP layer
        d_mlp = expanison_factor * d_model
        self.mlp_hidden = nn.Linear(d_model, d_mlp)
        self.mlp_second = nn.Linear(d_mlp, d_model)

        # self.d_head = d_head

    # def attention(self, query: LanguageInputType, key_value: VisionInputType) -> AttentionType:
    #     """Attention

    #     Attention(Q,K,V) = softmax( (Q K^T) / (sqrt(d_head)) ) * V"""
    #     # Merge across frame dimension
    #     query_merged: LanguageInputType = rearrange(
    #         "batch frame image_pos d_model -> batch (frame image_pos) d_model", query)

    #     # Numerator
    #     key_transpose: TT["batch", "pos", "d_head"] = rearrange(
    #         key_value, "batch pos d_head -> batch d_head pos")
    #     numerator: AttentionType = query_merged @ key_transpose

    #     attention_pattern: AttentionType = numerator / self.d_head

    #     softmax_part: AttentionType = nn.functional.softmax(
    #         attention_pattern, dim=-1)

    #     return einsum("batch des src, batch src d_head -> batch des src", softmax_part, key_value)

    def forward(self,
                vision_input: VisionInputType,
                language_input: LanguageInputType):
        """Applies a Gated Attention Layer"""
        # Q, K, V

        # Merge the vision input across frames
        vision: LanguageInputType = rearrange(
            "batch frame image_pos d_model -> batch (frame image_pos) d_model", vision_input)

        # Attention
        query = self.query_layer(language_input)
        key = self.key_layer(vision)
        value = self.value_layer(vision)
        attention_output, _attention_output_weights = self.attention(
            query, key, value)

        # 1. Gated Cross Attention
        y = vision_input + self.alpha_xattn.tanh() * attention_output

        # 2. Gated Feed Forward (dense) Layer
        ffw = self.mlp_second(self.mlp_hidden(y))
        y = y + self.alpha_dense.tanh() * ffw

        return y
