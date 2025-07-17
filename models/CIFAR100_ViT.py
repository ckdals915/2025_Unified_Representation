'''
******************************************************************************
* @author   Handong Global University Industrial Intelligence Lab
* @Mod	    2025-07-15
* @brief    Unified Representation Vision-Transformer(ViT) for CIFAR100 Dataset
******************************************************************************
'''

import torch
import torch.nn as nn
from torchinfo import summary

from modules import LinearAsGNN
from modules import Conv2dAsGNN
from modules import TransformerEncoderLayerAsGNN, TransformerEncoderAsGNN


# Standard ViT
class PatchEmbedding(nn.Module):
    """
    Patch Embedding for ViT:
      - Input: (B, 3, H, W)
      - Using Conv2D -> (B, num_patches, embed_dim)
    """
    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 8,
                 in_ch: int = 3,
                 embed_dim: int = 128,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.device = device

        # Projection layer: Conv2D
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size).to(self.device)

        # Class token (learnable) + Position Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, device=self.device))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim, device=self.device))

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Data
        x = x.to(self.device)

        # B: batch size
        B = x.size(0)
        # Patch Embedding
        x = self.proj(x)  # (B, embed_dim, 4, 4)

        # Flatten spatial dimensions
        x = x.flatten(2)
        
        # Transpose 
        x = x.transpose(1, 2) 

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)  

        # Add position embedding
        x = x + self.pos_embed
        return x

# Standard ViT Model
class ViTCIFAR100(nn.Module):
    """
      - patch embedding: Conv2d → flatten
      - nn.TransformerEncoderLayer / TransformerEncoder
      - classification head: [CLS]
    """
    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 8,
                 in_ch: int = 3,
                 num_classes: int = 100,
                 embed_dim: int = 128,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_ch=in_ch,
            embed_dim=embed_dim,
            device=self.device
        )
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        
        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer.to(self.device),
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim).to(self.device)
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim).to(self.device)
        self.head = nn.Linear(embed_dim, num_classes).to(self.device)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Data
        x = x.to(self.device)
        
        # patch embedding
        x = self.patch_embed(x)  

        # ransformer Encoder
        x = self.encoder(x) 

        # [CLS]
        cls_token = x[:, 0]  
        cls_token = self.norm(cls_token)
        logits = self.head(cls_token) 
        return logits



# ViTAsGNN
class PatchEmbeddingGNN(nn.Module):
    """
    ViTAsGNN의 Patch Embedding:
      - Conv2dAsGNN kernel 8x8
      - (B, embed_dim, H', W') → flatten → sequence
    """
    def __init__(self, 
                 img_size: int = 32, 
                 patch_size: int = 8, 
                 in_ch: int = 3, 
                 embed_dim: int = 128, 
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        
        # Parameters Configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.device=device

        # Patch Embedding using Conv2dAsGNN
        self.proj_gnn = Conv2dAsGNN(
            height=img_size,
            width=img_size,
            kernel_size=patch_size,
            stride=patch_size,
            in_channels=in_ch,
            out_channels=embed_dim,
            padding='valid',
            bias=True
        ).to(self.device)


        # Class token + Position Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim, device=self.device))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim, device=self.device))

        # Initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Data
        x = x.to(self.device)

        # B: batch size
        B = x.size(0)
        
        # Patch Embedding
        x = self.proj_gnn(x)  # (B, embed_dim, 4, 4)

        # flatten spatial
        x = x.flatten(2)
        
        # Transpose
        x = x.transpose(1, 2)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Poisition Embedding
        x = x + self.pos_embed
        return x

# ViTAsGNN Model
class ViTAsGNNCIFAR100(nn.Module):
    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 8,
                 in_ch: int = 3,
                 num_classes: int = 100,
                 embed_dim: int = 128,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device

        # Patch Embedding using Conv2dAsGNN
        self.patch_embed = PatchEmbeddingGNN(
            img_size=img_size,
            patch_size=patch_size,
            in_ch=in_ch,
            embed_dim=embed_dim,
            device=self.device
        )

        # Transformer Encoder Layer using TransformerEncoderLayerAsGNN
        encoder_layer_gnn = TransformerEncoderLayerAsGNN(
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            bias=True,
            batch_first=True,
            device=self.device
        )
        
        # Transformer Encoder using TransformerEncoderAsGNN
        self.encoder = TransformerEncoderAsGNN(
            encoder_layer=encoder_layer_gnn,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim).to(self.device)
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim).to(self.device)
        self.head = LinearAsGNN(embed_dim, num_classes, bias=True).to(self.device)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Data
        x = x.to(self.device)
        
        # Patch Embedding
        x = self.patch_embed(x)

        # Transformer Encoder
        x = self.encoder(x) 

        # Classification head
        cls_token = x[:, 0]    
        cls_token = self.norm(cls_token)
        logits = self.head(cls_token) 
        return logits



if __name__ == "__main__":
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    # ViT and ViTAsGNN Model Initialization
    vit_model = ViTCIFAR100(img_size=32, patch_size=8, in_ch=3, embed_dim=128,
                    depth=6, num_heads=8, mlp_ratio=4.0,
                    dropout=0.1, num_classes=100,
                    device=device).to(device)
    vit_as_gnn_model = ViTAsGNNCIFAR100(img_size=32, patch_size=8, in_ch=3, embed_dim=128,
                                depth=6, num_heads=8, mlp_ratio=4.0,
                                dropout=0.1, num_classes=100,
                                device=device).to(device)
    
    # Patch Embedding Summary
    print("\n\n--- [PatchEmbedding] Summary (표준 Conv2d) ---")
    summary(vit_model.patch_embed,
            input_size=(4, 3, 32, 32),
            device=device.type)

    print("\n\n--- [PatchEmbeddingGNN] Summary (Conv2dAsGNN) ---")
    summary(vit_as_gnn_model.patch_embed,
            input_size=(4, 3, 32, 32),
            device=device.type)

    try:
        print("\n\n--- [ViT 전체] Summary (시도) ---")
        summary(vit_model,
                input_size=(4, 3, 32, 32),
                device=device.type)
    except Exception as e:
        print("전체 ViT 구조 요약 중 오류 발생:", e)

    try:
        print("\n\n--- [ViTAsGNN 전체] Summary (시도) ---")
        summary(vit_as_gnn_model,
                input_size=(4, 3, 32, 32),
                device=device.type)
    except Exception as e:
        print("전체 ViTAsGNN 구조 요약 중 오류 발생:", e)