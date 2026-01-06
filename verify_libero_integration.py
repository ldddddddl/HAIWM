#!/usr/bin/env python3
"""
Verification script for LIBERO integration in H-AIF.
Tests all the new modules and modifications.
"""

import sys
import traceback

def test_imports():
    """Test that all new modules can be imported."""
    print("=" * 50)
    print("Testing imports...")
    print("=" * 50)
    
    errors = []
    
    # Test language encoder
    try:
        from model.language_encoder import CLIPLanguageEncoder, LanguageConditionedFusion
        print("[OK] CLIPLanguageEncoder imported")
    except Exception as e:
        errors.append(f"CLIPLanguageEncoder: {e}")
        print(f"[FAIL] CLIPLanguageEncoder: {e}")
    
    # Test libero dataset loader
    try:
        from load_libero_dataset import (
            LiberoDataset, 
            load_libero_dataloader, 
            denormalize_action,
            LiberoNormalizer,
            download_libero_dataset,
        )
        print("[OK] load_libero_dataset imported")
    except Exception as e:
        errors.append(f"load_libero_dataset: {e}")
        print(f"[FAIL] load_libero_dataset: {e}")
    
    # Test robosuite env
    try:
        from envs.robosuite_env import RobosuiteEnvWrapper, PolicyEvaluator
        print("[OK] robosuite_env imported")
    except Exception as e:
        errors.append(f"robosuite_env: {e}")
        print(f"[FAIL] robosuite_env: {e}")
    
    # Test models with language support
    try:
        from model.models import ActNet, MultiModalFusionModel, MultiModalAttention
        print("[OK] model.models imported")
    except Exception as e:
        errors.append(f"model.models: {e}")
        print(f"[FAIL] model.models: {e}")
    
    # Test decoders
    try:
        from model.decoders import ActGenerate, SuckerAct
        print("[OK] model.decoders imported")
    except Exception as e:
        errors.append(f"model.decoders: {e}")
        print(f"[FAIL] model.decoders: {e}")
    
    return errors


def test_clip_encoder():
    """Test CLIP language encoder functionality."""
    print("\n" + "=" * 50)
    print("Testing CLIP Language Encoder...")
    print("=" * 50)
    
    try:
        import torch
        from model.language_encoder import CLIPLanguageEncoder
        
        encoder = CLIPLanguageEncoder(
            clip_model="ViT-B/32",
            output_dim=120,
            freeze_clip=True,
        )
        
        texts = ["pick up the red cube", "place on the plate"]
        embeddings = encoder(texts)
        
        print(f"  Input: {texts}")
        print(f"  Output shape: {embeddings.shape}")
        assert embeddings.shape == (2, 120), f"Expected (2, 120), got {embeddings.shape}"
        print("[OK] CLIP encoder works correctly")
        return None
    except Exception as e:
        print(f"[FAIL] CLIP encoder: {e}")
        return str(e)


def test_config():
    """Test LIBERO config file."""
    print("\n" + "=" * 50)
    print("Testing config_libero.yaml...")
    print("=" * 50)
    
    try:
        from omegaconf import OmegaConf
        
        cfg = OmegaConf.load("config_libero.yaml")
        
        # Check key settings
        assert cfg.action_dim == 7, f"Expected action_dim=7, got {cfg.action_dim}"
        assert cfg.use_language == True, f"Expected use_language=True"
        assert cfg.clip_model == "ViT-B/32", f"Expected clip_model='ViT-B/32'"
        
        print(f"  action_dim: {cfg.action_dim}")
        print(f"  use_language: {cfg.use_language}")
        print(f"  clip_model: {cfg.clip_model}")
        print(f"  task_suite: {cfg.task_suite}")
        print("[OK] config_libero.yaml is valid")
        return None
    except Exception as e:
        print(f"[FAIL] Config: {e}")
        return str(e)


def test_normalizer():
    """Test LiberoNormalizer."""
    print("\n" + "=" * 50)
    print("Testing LiberoNormalizer...")
    print("=" * 50)
    
    try:
        import torch
        import numpy as np
        from load_libero_dataset import LiberoNormalizer
        
        # Create normalizer with dummy stats
        norm_stats = {
            "action": {
                "mean": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
                "std": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            }
        }
        normalizer = LiberoNormalizer(norm_stats=norm_stats)
        
        # Test normalize
        action = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        normalized = normalizer.normalize(action, "action")
        
        # Test denormalize
        denormalized = normalizer.denormalize(normalized, "action")
        
        # Check roundtrip
        assert torch.allclose(action, denormalized, atol=1e-5), "Roundtrip failed"
        
        print(f"  Original: {action}")
        print(f"  Normalized: {normalized}")
        print(f"  Denormalized: {denormalized}")
        print("[OK] LiberoNormalizer works correctly")
        return None
    except Exception as e:
        print(f"[FAIL] Normalizer: {e}")
        traceback.print_exc()
        return str(e)


def test_model_instantiation():
    """Test ActNet instantiation with LIBERO config."""
    print("\n" + "=" * 50)
    print("Testing ActNet instantiation...")
    print("=" * 50)
    
    try:
        import torch
        from omegaconf import OmegaConf
        from model.models import ActNet
        
        cfg = OmegaConf.load("config_libero.yaml")
        
        # Create model
        model = ActNet(cfg, is_use_cuda=False, device='cpu')
        
        # Check language encoder
        has_lang = hasattr(model, 'language_encoder') and model.language_encoder is not None
        print(f"  Language encoder present: {has_lang}")
        print(f"  use_language flag: {model.use_language}")
        print(f"  action_dim: {cfg.action_dim}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print("[OK] ActNet instantiated successfully")
        return None
    except Exception as e:
        print(f"[FAIL] Model instantiation: {e}")
        traceback.print_exc()
        return str(e)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  H-AIF LIBERO Integration Verification")
    print("=" * 60)
    
    all_errors = []
    
    # Test imports
    errors = test_imports()
    all_errors.extend(errors)
    
    # Test CLIP encoder
    error = test_clip_encoder()
    if error:
        all_errors.append(f"CLIP encoder: {error}")
    
    # Test config
    error = test_config()
    if error:
        all_errors.append(f"Config: {error}")
    
    # Test normalizer
    error = test_normalizer()
    if error:
        all_errors.append(f"Normalizer: {error}")
    
    # Test model instantiation
    error = test_model_instantiation()
    if error:
        all_errors.append(f"Model: {error}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    if all_errors:
        print(f"\n[FAILURES] {len(all_errors)} error(s):")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
