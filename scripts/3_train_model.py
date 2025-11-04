"""
Model Training Orchestration Script

This script orchestrates the complete training pipeline:
1. Environment validation (GPU, CUDA, disk space)
2. Data validation (training files exist)
3. SFT training with QLoRA (4-5 hours)
4. Optional DPO training (1-2 hours)
5. LoRA weight merging
6. GGUF conversion and quantization
7. Model verification

Hardware Requirements:
- RTX 3070 8GB or better
- ~30GB free disk space
- CUDA 11.8+ with PyTorch 2.0+

Usage:
    # SFT only (5 hours)
    python scripts/3_train_model.py --mode sft

    # SFT + DPO (6-7 hours)
    python scripts/3_train_model.py --mode sft+dpo

    # DPO only (1-2 hours, requires SFT checkpoint)
    python scripts/3_train_model.py --mode dpo --sft_checkpoint checkpoints/sft/final

    # Custom hyperparameters
    python scripts/3_train_model.py --mode sft --epochs 3 --batch_size 1 --lora_r 32

    # Test mode (small dataset, 1 epoch)
    python scripts/3_train_model.py --mode sft --test

    # Dry run (validation only)
    python scripts/3_train_model.py --mode sft --dry_run
"""

import argparse
import os
import sys
import shutil
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING]  PyTorch not installed")

from model.trainer import (
    load_base_model,
    setup_lora,
    load_training_data,
    train_sft,
    train_dpo,
    load_dpo_data,
    merge_and_save,
    check_dependencies
)


def validate_environment() -> Dict[str, Any]:
    """
    Validate training environment

    Checks:
    - PyTorch installed
    - CUDA available
    - GPU memory sufficient
    - Disk space available
    - Required libraries installed

    Returns:
        Dictionary with environment info

    Raises:
        RuntimeError if environment invalid
    """
    print(f"\n{'='*60}")
    print(f"[CHECK] ENVIRONMENT VALIDATION")
    print(f"{'='*60}\n")

    env_info = {}

    # Check PyTorch
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Install with: pip install torch")
    print(f"[OK] PyTorch: {torch.__version__}")
    env_info['pytorch_version'] = torch.__version__

    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU required for training.")
    print(f"[OK] CUDA: {torch.version.cuda}")
    env_info['cuda_version'] = torch.version.cuda

    # Check GPU
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"[OK] GPU: {gpu_name}")
    print(f"   Memory: {gpu_mem_gb:.1f} GB")
    print(f"   Count: {gpu_count}")

    env_info['gpu_name'] = gpu_name
    env_info['gpu_memory_gb'] = gpu_mem_gb
    env_info['gpu_count'] = gpu_count

    # Warn if GPU memory < 8GB
    if gpu_mem_gb < 7.5:
        print(f"\n[WARNING]  Warning: GPU has {gpu_mem_gb:.1f} GB memory")
        print(f"   Training may fail with batch_size=2")
        print(f"   Recommend: Use --batch_size 1 or upgrade to RTX 3070+")

    # Check disk space
    disk_usage = shutil.disk_usage(os.getcwd())
    free_gb = disk_usage.free / (1024 ** 3)
    print(f"[OK] Disk space: {free_gb:.1f} GB free")

    env_info['disk_free_gb'] = free_gb

    if free_gb < 30:
        print(f"\n[WARNING]  Warning: Only {free_gb:.1f} GB free")
        print(f"   Training requires ~30GB for checkpoints")
        print(f"   Consider cleaning up space before training")

    # Check dependencies
    try:
        check_dependencies()
        print(f"[OK] Dependencies: unsloth, trl installed")
    except ImportError as e:
        raise RuntimeError(f"Missing dependencies: {e}")

    print(f"\n{'='*60}")
    print(f"[OK] Environment validation passed!")
    print(f"{'='*60}\n")

    return env_info


def validate_training_data(
    train_path: str,
    val_path: Optional[str] = None,
    dpo_path: Optional[str] = None,
    mode: str = "sft"
) -> Dict[str, int]:
    """
    Validate training data files exist and count examples

    Args:
        train_path: Path to training data
        val_path: Path to validation data
        dpo_path: Path to DPO pairs
        mode: Training mode ("sft", "sft+dpo", "dpo")

    Returns:
        Dictionary with example counts

    Raises:
        FileNotFoundError if required files missing
    """
    print(f"\n{'='*60}")
    print(f"[DATA] DATA VALIDATION")
    print(f"{'='*60}\n")

    counts = {}

    # Training data (required for SFT)
    if mode in ["sft", "sft+dpo"]:
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")

        train_count = sum(1 for _ in open(train_path, 'r', encoding='utf-8'))
        print(f"[OK] Training data: {train_count:,} examples")
        print(f"   Path: {train_path}")
        counts['train'] = train_count

        if train_count < 100:
            print(f"\n[WARNING]  Warning: Only {train_count} training examples")
            print(f"   Recommend at least 1,000 for quality results")

    # Validation data (optional but recommended)
    if val_path and os.path.exists(val_path):
        val_count = sum(1 for _ in open(val_path, 'r', encoding='utf-8'))
        print(f"[OK] Validation data: {val_count:,} examples")
        print(f"   Path: {val_path}")
        counts['val'] = val_count
    else:
        print(f"[WARNING]  No validation data (evaluation disabled)")
        counts['val'] = 0

    # DPO data (required for DPO training)
    if mode in ["sft+dpo", "dpo"] and dpo_path:
        if not os.path.exists(dpo_path):
            raise FileNotFoundError(f"DPO data not found: {dpo_path}")

        dpo_count = sum(1 for _ in open(dpo_path, 'r', encoding='utf-8'))
        print(f"[OK] DPO pairs: {dpo_count:,} pairs")
        print(f"   Path: {dpo_path}")
        counts['dpo'] = dpo_count

        if dpo_count < 100:
            print(f"\n[WARNING]  Warning: Only {dpo_count} DPO pairs")
            print(f"   Recommend at least 500 for meaningful DPO training")

    print(f"\n{'='*60}")
    print(f"[OK] Data validation passed!")
    print(f"{'='*60}\n")

    return counts


def run_sft_training(
    config: Dict[str, Any],
    env_info: Dict[str, Any]
) -> str:
    """
    Run SFT training phase

    Args:
        config: Training configuration
        env_info: Environment information

    Returns:
        Path to SFT checkpoint
    """
    print(f"\n{'='*60}")
    print(f"==> PHASE 1: SUPERVISED FINE-TUNING (SFT)")
    print(f"{'='*60}\n")

    # Load base model
    model, tokenizer = load_base_model(
        model_name=config['base_model'],
        max_seq_length=config['max_seq_length'],
        load_in_4bit=True
    )

    # Setup LoRA
    model = setup_lora(
        model,
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout']
    )

    # Load training data
    train_dataset, eval_dataset = load_training_data(
        train_path=config['train_data'],
        val_path=config['val_data'],
        max_samples=config.get('max_samples')
    )

    # Inform user about validation strategy
    print(f"\n📊 Validation Strategy:")
    print(f"   Training evaluation: {config['val_data']}")
    if 'val_sft_train' in config['val_data']:
        print(f"   └─ Fast evaluation (~2 minutes per checkpoint)")
        print(f"   After training: Use scripts/evaluate_checkpoints.py for full evaluation")
    print()

    # Train SFT
    model = train_sft(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=config['sft_output_dir'],
        num_train_epochs=config['sft_epochs'],
        learning_rate=config['sft_learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        warmup_ratio=config['warmup_ratio'],
        max_seq_length=config['max_seq_length'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        seed=config['seed'],
        report_to=config['report_to']
    )

    sft_checkpoint = f"{config['sft_output_dir']}/final"
    print(f"\n[OK] SFT training complete: {sft_checkpoint}")

    return sft_checkpoint


def run_dpo_training(
    sft_checkpoint: str,
    config: Dict[str, Any]
) -> str:
    """
    Run DPO training phase

    Args:
        sft_checkpoint: Path to SFT checkpoint
        config: Training configuration

    Returns:
        Path to DPO checkpoint
    """
    print(f"\n{'='*60}")
    print(f"💎 PHASE 2: DIRECT PREFERENCE OPTIMIZATION (DPO)")
    print(f"{'='*60}\n")

    # Load SFT model
    print(f"Loading SFT checkpoint: {sft_checkpoint}")
    model, tokenizer = load_base_model(
        model_name=sft_checkpoint,
        max_seq_length=config['max_seq_length'],
        load_in_4bit=True
    )

    # Setup LoRA (if not already applied)
    # Note: If loading from checkpoint, LoRA may already be in model
    # This is safe to call again

    # Load DPO data
    dpo_dataset = load_dpo_data(
        dpo_path=config['dpo_data'],
        max_samples=config.get('max_samples')
    )

    # Train DPO
    model = train_dpo(
        model=model,
        tokenizer=tokenizer,
        dpo_dataset=dpo_dataset,
        output_dir=config['dpo_output_dir'],
        beta=config['dpo_beta'],
        num_train_epochs=config['dpo_epochs'],
        learning_rate=config['dpo_learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'] // 2,
        max_seq_length=config['max_seq_length'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        seed=config['seed'],
        report_to=config['report_to']
    )

    dpo_checkpoint = f"{config['dpo_output_dir']}/final"
    print(f"\n[OK] DPO training complete: {dpo_checkpoint}")

    return dpo_checkpoint


def merge_and_save_final(
    checkpoint_path: str,
    output_dir: str
) -> str:
    """
    Merge LoRA weights and save final model

    Args:
        checkpoint_path: Path to final checkpoint (SFT or DPO)
        output_dir: Output directory for merged model

    Returns:
        Path to merged model
    """
    print(f"\n{'='*60}")
    print(f"[SAVE] PHASE 3: MERGING & SAVING")
    print(f"{'='*60}\n")

    # Load model
    model, tokenizer = load_base_model(
        model_name=checkpoint_path,
        max_seq_length=2048,
        load_in_4bit=True
    )

    # Merge and save
    merged_path = merge_and_save(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        save_method="merged_16bit"
    )

    print(f"\n[OK] Model merged and saved: {merged_path}")

    return merged_path


def convert_to_gguf(
    hf_model_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M"
) -> str:
    """
    Convert HuggingFace model to GGUF and quantize

    Requires llama.cpp to be available.

    Args:
        hf_model_path: Path to HuggingFace model
        output_path: Output path for GGUF file
        quant_type: Quantization type (Q4_K_M recommended)

    Returns:
        Path to quantized GGUF file
    """
    print(f"\n{'='*60}")
    print(f"🔧 PHASE 4: GGUF CONVERSION & QUANTIZATION")
    print(f"{'='*60}\n")

    print(f"[WARNING]  GGUF conversion requires llama.cpp")
    print(f"   Manual steps:")
    print(f"   1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
    print(f"   2. Build: cd llama.cpp && make")
    print(f"   3. Convert to F16: python convert.py {hf_model_path}")
    print(f"   4. Quantize to {quant_type}: ./quantize {hf_model_path}/ggml-model-f16.gguf {output_path} {quant_type}")
    print(f"\n   Or use the automated script if available")

    # TODO: Implement automated GGUF conversion if llama.cpp Python bindings available
    # For now, print manual instructions

    print(f"\n   Target output: {output_path}")
    print(f"   Quantization: {quant_type}")
    print(f"   Expected size: ~2.2 GB")

    return output_path


def print_training_summary(
    config: Dict[str, Any],
    env_info: Dict[str, Any],
    data_counts: Dict[str, int],
    final_model_path: str,
    start_time: datetime,
    end_time: datetime
):
    """
    Print comprehensive training summary

    Args:
        config: Training configuration
        env_info: Environment information
        data_counts: Data example counts
        final_model_path: Path to final model
        start_time: Training start time
        end_time: Training end time
    """
    duration = end_time - start_time
    hours = duration.total_seconds() / 3600

    print(f"\n{'='*60}")
    print(f"[OK] TRAINING COMPLETE!")
    print(f"{'='*60}\n")

    print(f"Training Summary:")
    print(f"  Mode: {config['mode']}")
    print(f"  Base model: {config['base_model']}")
    print(f"  Training data: {data_counts.get('train', 0):,} examples")
    if config['mode'] in ["sft+dpo", "dpo"]:
        print(f"  DPO pairs: {data_counts.get('dpo', 0):,} pairs")
    print(f"  Duration: {hours:.1f} hours")

    print(f"\nHyperparameters:")
    print(f"  LoRA rank: {config['lora_r']}")
    print(f"  LoRA alpha: {config['lora_alpha']}")
    print(f"  SFT epochs: {config['sft_epochs']}")
    if config['mode'] in ["sft+dpo", "dpo"]:
        print(f"  DPO epochs: {config['dpo_epochs']}")
        print(f"  DPO beta: {config['dpo_beta']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")

    print(f"\nHardware:")
    print(f"  GPU: {env_info['gpu_name']}")
    print(f"  VRAM: {env_info['gpu_memory_gb']:.1f} GB")

    print(f"\nOutput:")
    print(f"  Final model: {final_model_path}")

    print(f"\nNext Steps:")
    print(f"  1. Convert to GGUF Q4_K_M for inference (manual)")
    print(f"  2. Evaluate: python scripts/4_evaluate_personality.py")
    print(f"  3. Deploy: Update .env MODEL_PATH and start bot")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train personality model with QLoRA + DPO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode
    parser.add_argument(
        '--mode',
        choices=['sft', 'sft+dpo', 'dpo'],
        default='sft',
        help='Training mode (default: sft)'
    )

    # Data paths
    parser.add_argument(
        '--train_data',
        default='data_storage/training/train_sft.jsonl',
        help='Training data path'
    )
    parser.add_argument(
        '--val_data',
        default='data_storage/training/val_sft_train.jsonl',
        help='Validation data path (use val_sft_train.jsonl for fast eval during training, val_sft_full.jsonl for comprehensive final eval)'
    )
    parser.add_argument(
        '--dpo_data',
        default='data_storage/training/dpo_pairs.jsonl',
        help='DPO pairs data path'
    )

    # Model
    parser.add_argument(
        '--base_model',
        default='Qwen/Qwen2.5-3B-Instruct',
        help='Base model name or path'
    )
    parser.add_argument(
        '--sft_checkpoint',
        default=None,
        help='SFT checkpoint path (for dpo mode)'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        default='models/finetuned/qwen2.5-3b-personality',
        help='Output directory for final model'
    )
    parser.add_argument(
        '--sft_output_dir',
        default='checkpoints/sft',
        help='SFT checkpoints directory'
    )
    parser.add_argument(
        '--dpo_output_dir',
        default='checkpoints/dpo',
        help='DPO checkpoints directory'
    )

    # Hyperparameters
    parser.add_argument('--sft_epochs', type=int, default=5, help='SFT epochs')
    parser.add_argument('--dpo_epochs', type=int, default=2, help='DPO epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU (1 for RTX 3070 8GB, 2 for 12GB+)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='Gradient accumulation (32 maintains effective batch=32 with batch_size=1)')
    parser.add_argument('--sft_learning_rate', type=float, default=1e-4, help='SFT learning rate')
    parser.add_argument('--dpo_learning_rate', type=float, default=5e-5, help='DPO learning rate')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank (16 for RTX 3070 8GB, 32 for 12GB+, 64 for 24GB+)')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha (typically 2x rank, so 32 when rank=16)')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--dpo_beta', type=float, default=0.1, help='DPO beta')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='Max sequence length (1024 covers 98% of data, 2048 for full coverage)')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio')

    # Logging
    parser.add_argument('--logging_steps', type=int, default=10, help='Log every N steps')
    parser.add_argument('--save_steps', type=int, default=500, help='Save every N steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='Eval every N steps')
    parser.add_argument('--report_to', default='none', help='Logging backend (none, wandb, tensorboard)')

    # Testing
    parser.add_argument('--test', action='store_true', help='Test mode (small dataset, 1 epoch)')
    parser.add_argument('--dry_run', action='store_true', help='Dry run (validation only)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Build config
    config = vars(args)

    # Test mode adjustments
    if args.test:
        print("\n[WARNING]  TEST MODE: Using small dataset and 1 epoch")
        config['sft_epochs'] = 1
        config['dpo_epochs'] = 1
        config['max_samples'] = 100
        config['save_steps'] = 50
        config['eval_steps'] = 50

    # Header
    print(f"\n{'='*60}")
    print(f"==> MODEL TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Base Model: {args.base_model}")
    print(f"Output: {args.output_dir}")
    if args.test:
        print(f"[WARNING]  TEST MODE ENABLED")
    if args.dry_run:
        print(f"[WARNING]  DRY RUN (validation only)")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    try:
        # Step 1: Validate environment
        env_info = validate_environment()

        # Step 2: Validate data
        data_counts = validate_training_data(
            train_path=args.train_data,
            val_path=args.val_data,
            dpo_path=args.dpo_data if args.mode in ["sft+dpo", "dpo"] else None,
            mode=args.mode
        )

        if args.dry_run:
            print("\n[OK] Dry run complete. Environment and data validated.")
            print("   Remove --dry_run to start training.")
            return

        # Step 3: Confirm training
        print(f"\n{'='*60}")
        print(f"[WARNING]  CONFIRM TRAINING")
        print(f"{'='*60}")
        print(f"This will take approximately:")
        if args.mode == "sft":
            print(f"  - SFT: 4-5 hours on RTX 3070")
        elif args.mode == "sft+dpo":
            print(f"  - SFT: 4-5 hours")
            print(f"  - DPO: 1-2 hours")
            print(f"  - Total: 6-7 hours")
        elif args.mode == "dpo":
            print(f"  - DPO: 1-2 hours")
        print(f"\nCheckpoints will be saved to: {args.sft_output_dir}")
        print(f"Final model will be saved to: {args.output_dir}")
        print(f"\n{'='*60}\n")

        response = input("Continue with training? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return

        # Step 4: Run training based on mode
        if args.mode == "sft":
            sft_checkpoint = run_sft_training(config, env_info)
            final_checkpoint = sft_checkpoint

        elif args.mode == "sft+dpo":
            sft_checkpoint = run_sft_training(config, env_info)
            dpo_checkpoint = run_dpo_training(sft_checkpoint, config)
            final_checkpoint = dpo_checkpoint

        elif args.mode == "dpo":
            if not args.sft_checkpoint:
                raise ValueError("--sft_checkpoint required for DPO-only mode")
            dpo_checkpoint = run_dpo_training(args.sft_checkpoint, config)
            final_checkpoint = dpo_checkpoint

        # Step 5: Merge and save
        final_model_path = merge_and_save_final(
            checkpoint_path=final_checkpoint,
            output_dir=args.output_dir
        )

        # Step 6: GGUF conversion (manual instructions)
        gguf_path = f"{args.output_dir}/qwen2.5-3b-personality-q4.gguf"
        convert_to_gguf(final_model_path, gguf_path)

        end_time = datetime.now()

        # Step 7: Print summary
        print_training_summary(
            config, env_info, data_counts,
            final_model_path, start_time, end_time
        )

    except KeyboardInterrupt:
        print("\n\n[WARNING]  Training interrupted by user")
        print("   Checkpoints have been saved")
        print("   You can resume by using the latest checkpoint")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n[FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
