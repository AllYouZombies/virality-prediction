import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F


class ViViT(nn.Module):
    def __init__(self,
                 image_size=240,
                 patch_size=16,
                 num_classes=2,
                 num_frames=134,
                 dim=128,
                 depth=6,
                 heads=8,
                 mlp_dim=256,
                 dropout=0.1):
        super().__init__()

        patch_height = patch_width = image_size // patch_size
        patch_dim = 3 * patch_size * patch_size
        self.num_patches = patch_height * patch_width * num_frames

        self.patch_embed = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b (t h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=depth
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        x = self.patch_embed(video)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        return self.mlp_head(x[:, 0])
