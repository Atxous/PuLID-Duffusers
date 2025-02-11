import torch
import torch.nn as nn
import math

from .utils import reshape_tensor

class IDEncoder(nn.Module):
    def __init__(self, width=1280, context_dim=2048, num_token=5):
        super().__init__()
        self.num_token = num_token
        self.context_dim = context_dim
        h1 = min((context_dim * num_token) // 4, 1024)
        h2 = min((context_dim * num_token) // 2, 1024)
        self.body = nn.Sequential(
            nn.Linear(width, h1),
            nn.LayerNorm(h1),
            nn.LeakyReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.LeakyReLU(),
            nn.Linear(h2, context_dim * num_token),
        )

        for i in range(5):
            setattr(
                self,
                f'mapping_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, context_dim),
                ),
            )

            setattr(
                self,
                f'mapping_patch_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, context_dim),
                ),
            )

    def forward(self, x, y):
        # x shape [N, C]
        x = self.body(x)
        x = x.reshape(-1, self.num_token, self.context_dim)

        hidden_states = ()
        for i, emb in enumerate(y):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(
                emb[:, 1:]
            ).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state,)
        hidden_states = torch.cat(hidden_states, dim=1)

        return torch.cat([x, hidden_states], dim=1)
    


class IDFormer(nn.Module):
    """
    - perceiver resampler like arch (compared with previous MLP-like arch)
    - we concat id embedding (generated by arcface) and query tokens as latents
    - latents will attend each other and interact with vit features through cross-attention
    - vit features are multi-scaled and inserted into IDFormer in order, currently, each scale corresponds to two
      IDFormer layers
    """
    def __init__(
            self,
            dim=1024,
            depth=10,
            dim_head=64,
            heads=16,
            num_id_token=5,
            num_queries=32,
            output_dim=2048,
            ff_mult=4,
    ):
        super().__init__()

        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = dim ** -0.5

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        for i in range(5):
            setattr(
                self,
                f'mapping_{i}',
                nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.LeakyReLU(),
                    nn.Linear(1024, dim),
                ),
            )

        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x, y):

        latents = self.latents.repeat(x.size(0), 1, 1)

        num_duotu = x.shape[1] if x.ndim == 3 else 1

        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token * num_duotu, self.dim)

        latents = torch.cat((latents, x), dim=1)

        for i in range(5):
            vit_feature = getattr(self, f'mapping_{i}')(y[i])
            ctx_feature = torch.cat((x, vit_feature), dim=1)
            for attn, ff in self.layers[i * self.depth: (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        latents = latents[:, :self.num_queries]
        latents = latents @ self.proj_out
        return latents
    

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )



class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)

        return self.to_out(out)