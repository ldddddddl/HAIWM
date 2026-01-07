#!/usr/bin/env python3
"""
Baseline Behavior Cloning Models for Comparison

This module provides simple BC-RNN and BC-Transformer baseline models
for comparison with the H-AIF framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BCRNN(nn.Module):
    """
    Behavior Cloning with RNN (BC-RNN).

    A simple baseline that uses CNN image encoders + RNN for action prediction.
    No Active Inference, no language conditioning.
    """

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 2,
        image_size: int = 112,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Image encoder (simple CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
        )

        # Feature fusion
        # 2 cameras + state
        fusion_dim = hidden_dim * 2 + hidden_dim // 2
        self.fusion = nn.Linear(fusion_dim, hidden_dim)

        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Gripper prediction (binary)
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode_image(self, img):
        """Encode a single image."""
        return self.image_encoder(img)

    def forward(self, batch, phase: str = ""):
        """
        Forward pass.

        Args:
            batch: Dict containing 'observation' with images and state
            phase: 'train' or 'eval'
        """
        # Get observations
        top_img = batch["observation"]["top_image"]
        wrist_img = batch["observation"]["wrist_image"]
        state = batch["observation"]["state"]

        # Ensure correct shape
        if top_img.dim() == 5:  # [B, T, C, H, W]
            B, T = top_img.shape[:2]
            top_img = top_img[:, -1]  # Take last frame
            wrist_img = wrist_img[:, -1]
        else:
            B = top_img.shape[0]

        # Encode images
        top_feat = self.encode_image(top_img)  # [B, hidden_dim]
        wrist_feat = self.encode_image(wrist_img)  # [B, hidden_dim]

        # Encode state
        state_feat = self.state_encoder(state)  # [B, hidden_dim // 2]

        # Fuse features
        fused = torch.cat([top_feat, wrist_feat, state_feat], dim=-1)
        fused = self.fusion(fused)  # [B, hidden_dim]

        # Add sequence dimension for RNN
        fused = fused.unsqueeze(1)  # [B, 1, hidden_dim]

        # RNN forward
        rnn_out, _ = self.rnn(fused)  # [B, 1, hidden_dim]
        rnn_out = rnn_out.squeeze(1)  # [B, hidden_dim]

        # Predict actions
        actions = self.action_head(rnn_out)  # [B, action_dim]
        gripper = self.gripper_head(rnn_out)  # [B, 1]

        # Create output container
        class Output:
            pass

        output = Output()
        output.actions = actions.unsqueeze(1)  # [B, 1, action_dim]
        output.sucker = gripper.unsqueeze(1)  # [B, 1, 1]
        output.z_mix = None
        output.attention_weights = None
        output.mu = None
        output.var = None

        return output


class BCTransformer(nn.Module):
    """
    Behavior Cloning with Transformer (BC-Transformer).

    A baseline that uses CNN image encoders + Transformer for action prediction.
    No Active Inference, no language conditioning.
    """

    def __init__(
        self,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        image_size: int = 112,
        dropout: float = 0.1,
        max_seq_len: int = 10,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Image encoder (same as BC-RNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
        )

        # Feature fusion
        fusion_dim = hidden_dim * 2 + hidden_dim // 2
        self.fusion = nn.Linear(fusion_dim, hidden_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Gripper prediction
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode_image(self, img):
        """Encode a single image."""
        return self.image_encoder(img)

    def forward(self, batch, phase: str = ""):
        """Forward pass."""
        # Get observations
        top_img = batch["observation"]["top_image"]
        wrist_img = batch["observation"]["wrist_image"]
        state = batch["observation"]["state"]

        # Handle sequence input
        if top_img.dim() == 5:  # [B, T, C, H, W]
            B, T = top_img.shape[:2]
            # Encode each frame
            top_feat_list = []
            wrist_feat_list = []
            for t in range(T):
                top_feat_list.append(self.encode_image(top_img[:, t]))
                wrist_feat_list.append(self.encode_image(wrist_img[:, t]))
            top_feat = torch.stack(top_feat_list, dim=1)  # [B, T, hidden_dim]
            wrist_feat = torch.stack(wrist_feat_list, dim=1)  # [B, T, hidden_dim]
            state_feat = self.state_encoder(state).unsqueeze(1).expand(-1, T, -1)
        else:
            B = top_img.shape[0]
            top_feat = self.encode_image(top_img).unsqueeze(1)  # [B, 1, hidden_dim]
            wrist_feat = self.encode_image(wrist_img).unsqueeze(1)
            state_feat = self.state_encoder(state).unsqueeze(1)
            T = 1

        # Fuse features
        fused = torch.cat([top_feat, wrist_feat, state_feat], dim=-1)
        fused = self.fusion(fused)  # [B, T, hidden_dim]

        # Add positional encoding
        fused = fused + self.pos_embedding[:, :T, :]

        # Transformer forward
        trans_out = self.transformer(fused)  # [B, T, hidden_dim]

        # Take last timestep for prediction
        last_out = trans_out[:, -1, :]  # [B, hidden_dim]

        # Predict actions
        actions = self.action_head(last_out)  # [B, action_dim]
        gripper = self.gripper_head(last_out)  # [B, 1]

        # Create output container
        class Output:
            pass

        output = Output()
        output.actions = actions.unsqueeze(1)  # [B, 1, action_dim]
        output.sucker = gripper.unsqueeze(1)  # [B, 1, 1]
        output.z_mix = None
        output.attention_weights = None
        output.mu = None
        output.var = None

        return output


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing BC-RNN...")
    model = BCRNN(action_dim=7, hidden_dim=256)

    batch = {
        "observation": {
            "top_image": torch.randn(2, 3, 112, 112),
            "wrist_image": torch.randn(2, 3, 112, 112),
            "state": torch.randn(2, 7),
        }
    }

    output = model(batch)
    print(f"BC-RNN output actions shape: {output.actions.shape}")
    print(f"BC-RNN output gripper shape: {output.sucker.shape}")

    print("\nTesting BC-Transformer...")
    model = BCTransformer(action_dim=7, hidden_dim=256)

    output = model(batch)
    print(f"BC-Transformer output actions shape: {output.actions.shape}")
    print(f"BC-Transformer output gripper shape: {output.sucker.shape}")

    print("\nAll tests passed!")
