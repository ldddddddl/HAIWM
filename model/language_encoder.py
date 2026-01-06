"""
CLIP Language Encoder for H-AIF

This module provides a CLIP-based language encoder for encoding
task descriptions in the LIBERO benchmark.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union

# Try different import paths for CLIP
_CLIP_BACKEND = None

try:
    import clip

    _CLIP_BACKEND = "openai"
except ImportError:
    pass

if _CLIP_BACKEND is None:
    try:
        from transformers import CLIPTextModel, CLIPTokenizer

        _CLIP_BACKEND = "transformers"
    except ImportError:
        pass


class CLIPLanguageEncoder(nn.Module):
    """
    CLIP-based language encoder using ViT-B/32.

    Encodes text instructions into embeddings that can be fused
    with visual and proprioceptive features.

    Args:
        clip_model: CLIP model variant (default: "ViT-B/32")
        embed_dim: CLIP embedding dimension (512 for ViT-B/32)
        output_dim: Output dimension after projection
        freeze_clip: Whether to freeze CLIP weights
        device: Device to load the model on
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        embed_dim: int = 512,
        output_dim: int = 120,
        freeze_clip: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.clip_model_name = clip_model
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.freeze_clip = freeze_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._backend = _CLIP_BACKEND

        if self._backend is None:
            print("[WARNING] Neither openai-clip nor transformers CLIP available.")
            print(
                "[WARNING] Using dummy encoder. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
            self._use_dummy = True
        else:
            self._use_dummy = False
            self._load_clip_model()

        # Projection layer: CLIP embedding -> output dimension
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Cache for encoded texts (to avoid re-encoding same instructions)
        self._cache = {}
        self._cache_max_size = 1000

        # Move projection layer to target device
        self.projection = self.projection.to(self.device)

    def _load_clip_model(self):
        """Load the CLIP model based on available backend."""
        if self._backend == "openai":
            self._load_openai_clip()
        elif self._backend == "transformers":
            self._load_transformers_clip()

    def _load_openai_clip(self):
        """Load OpenAI CLIP model."""
        print(f"[CLIP] Loading OpenAI CLIP model: {self.clip_model_name}")
        self.clip, self.preprocess = clip.load(
            self.clip_model_name, device=self.device, jit=False
        )

        if self.freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
            self.clip.eval()

        self.tokenize = clip.tokenize

    def _load_transformers_clip(self):
        """Load HuggingFace Transformers CLIP model."""
        print(f"[CLIP] Loading Transformers CLIP model")
        model_name = "openai/clip-vit-base-patch32"

        self.clip = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        self.clip = self.clip.to(self.device)

        if self.freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
            self.clip.eval()

    def _encode_text_openai(self, texts: List[str]) -> torch.Tensor:
        """Encode text using OpenAI CLIP."""
        tokens = self.tokenize(texts, truncate=True).to(self.device)

        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            features = self.clip.encode_text(tokens)

        return features.float()

    def _encode_text_transformers(self, texts: List[str]) -> torch.Tensor:
        """Encode text using Transformers CLIP."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=77,
        ).to(self.device)

        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            outputs = self.clip(**inputs)
            features = outputs.pooler_output

        return features.float()

    def _encode_text_dummy(self, texts: List[str]) -> torch.Tensor:
        """Return dummy embeddings when CLIP is not available."""
        batch_size = len(texts)
        return torch.randn(batch_size, self.embed_dim, device=self.device)

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text(s) into CLIP embeddings.

        Args:
            texts: Single text or list of texts

        Returns:
            Tensor of shape [batch_size, embed_dim]
        """
        if isinstance(texts, str):
            texts = [texts]

        if self._use_dummy:
            return self._encode_text_dummy(texts)
        elif self._backend == "openai":
            return self._encode_text_openai(texts)
        elif self._backend == "transformers":
            return self._encode_text_transformers(texts)
        else:
            return self._encode_text_dummy(texts)

    def forward(
        self,
        texts: Union[str, List[str]],
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Encode texts and project to output dimension.

        Args:
            texts: Single text or list of texts
            use_cache: Whether to use caching for repeated texts

        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        if isinstance(texts, str):
            texts = [texts]

        # Check cache
        if use_cache:
            cache_hits = []
            cache_misses = []
            cache_miss_indices = []

            for i, text in enumerate(texts):
                if text in self._cache:
                    cache_hits.append((i, self._cache[text]))
                else:
                    cache_misses.append(text)
                    cache_miss_indices.append(i)

            if len(cache_misses) == 0:
                # All texts are cached
                embeddings = torch.stack([emb for _, emb in sorted(cache_hits)])
                return embeddings.to(self.device)

            # Encode cache misses
            if len(cache_misses) > 0:
                clip_embeddings = self.encode_text(cache_misses)
                projected = self.projection(clip_embeddings)

                # Update cache
                for text, emb in zip(cache_misses, projected):
                    if len(self._cache) < self._cache_max_size:
                        self._cache[text] = emb.detach().cpu()

            # Combine cached and new embeddings
            if len(cache_hits) > 0:
                all_embeddings = [None] * len(texts)
                for i, emb in cache_hits:
                    all_embeddings[i] = emb.to(self.device)
                for i, emb in zip(cache_miss_indices, projected):
                    all_embeddings[i] = emb
                return torch.stack(all_embeddings)
            else:
                return projected
        else:
            # No caching
            clip_embeddings = self.encode_text(texts)
            return self.projection(clip_embeddings)

    def clear_cache(self):
        """Clear the text embedding cache."""
        self._cache.clear()

    @property
    def requires_grad(self) -> bool:
        """Check if the encoder requires gradients."""
        return not self.freeze_clip or any(
            p.requires_grad for p in self.projection.parameters()
        )


class LanguageConditionedFusion(nn.Module):
    """
    Module for fusing language embeddings with other modalities.

    Uses cross-attention to condition visual/action features
    on language instructions.
    """

    def __init__(
        self,
        lang_dim: int = 120,
        feature_dim: int = 240,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lang_dim = lang_dim
        self.feature_dim = feature_dim

        # Project language to match feature dimension
        self.lang_proj = nn.Linear(lang_dim, feature_dim)

        # Cross-attention: features attend to language
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        features: torch.Tensor,
        language: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse features with language conditioning.

        Args:
            features: [B, S, D] visual/action features
            language: [B, lang_dim] language embeddings

        Returns:
            Fused features [B, S, D]
        """
        B, S, D = features.shape

        # Project language and expand for cross-attention
        lang = self.lang_proj(language)  # [B, D]
        lang = lang.unsqueeze(1)  # [B, 1, D]

        # Cross-attention: features attend to language
        attn_out, _ = self.cross_attention(
            query=features,
            key=lang,
            value=lang,
        )

        # Residual connection and norm
        features = self.norm1(features + attn_out)

        # Feed-forward with residual
        features = self.norm2(features + self.ffn(features))

        return features


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing CLIPLanguageEncoder...")

    encoder = CLIPLanguageEncoder(
        clip_model="ViT-B/32",
        output_dim=120,
    )

    texts = [
        "pick up the red cube",
        "place the object on the plate",
        "open the drawer and take out the bowl",
    ]

    embeddings = encoder(texts)
    print(f"Input texts: {texts}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: ({len(texts)}, 120)")

    assert embeddings.shape == (len(texts), 120), f"Shape mismatch: {embeddings.shape}"
    print("Test passed!")

    # Test fusion module
    print("\nTesting LanguageConditionedFusion...")

    fusion = LanguageConditionedFusion(
        lang_dim=120,
        feature_dim=240,
    )

    features = torch.randn(2, 10, 240)
    language = torch.randn(2, 120)

    fused = fusion(features, language)
    print(f"Features shape: {features.shape}")
    print(f"Language shape: {language.shape}")
    print(f"Fused shape: {fused.shape}")

    assert fused.shape == features.shape, f"Shape mismatch: {fused.shape}"
    print("Test passed!")
