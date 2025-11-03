"""
Model Training Module - QLoRA + DPO for Personality Fine-Tuning

This module implements the core training logic for fine-tuning Qwen2.5-3B-Instruct
with personality using:
- QLoRA (Quantized Low-Rank Adaptation) for memory-efficient training
- SFT (Supervised Fine-Tuning) for base personality learning
- DPO (Direct Preference Optimization) for style reinforcement

Hardware Requirements:
- RTX 3070 8GB or better
- ~30GB disk space for checkpoints
- CUDA 11.8+ with PyTorch 2.0+

Training Time (RTX 3070):
- SFT: 4-5 hours (5 epochs)
- DPO: 1-2 hours (2 epochs)
- Total: 5-7 hours

Usage:
    from model.trainer import train_sft, train_dpo, load_base_model

    # Load base model
    model, tokenizer = load_base_model("Qwen/Qwen2.5-3B-Instruct")

    # Train SFT
    model = train_sft(
        model, tokenizer,
        train_dataset, eval_dataset,
        output_dir="checkpoints/sft"
    )

    # Train DPO (optional but recommended)
    model = train_dpo(
        model, tokenizer,
        dpo_dataset,
        output_dir="checkpoints/dpo"
    )

    # Save final model
    save_model_for_inference(model, tokenizer, "models/finetuned/personality")
"""

import os
import json
import torch
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Unsloth for efficient QLoRA training
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: Unsloth not installed. Install with:")
    print("   pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
    UNSLOTH_AVAILABLE = False

# TRL for SFT and DPO training
try:
    from trl import SFTTrainer, DPOTrainer, DPOConfig
    TRL_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: TRL not installed. Install with:")
    print("   pip install trl")
    TRL_AVAILABLE = False


def check_dependencies():
    """Check if required libraries are installed"""
    if not UNSLOTH_AVAILABLE:
        raise ImportError("Unsloth is required for training. See installation instructions above.")
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for training. Install with: pip install trl")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training. No GPU detected.")


def load_base_model(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    dtype: Optional[str] = None,
    attn_implementation: str = "eager"
) -> Tuple[Any, Any]:
    """
    Load base model with 4-bit quantization using Unsloth

    Args:
        model_name: HuggingFace model name or path
        max_seq_length: Maximum sequence length for training
        load_in_4bit: Use 4-bit quantization (required for RTX 3070)
        dtype: Data type (None = auto-detect)
        attn_implementation: Attention implementation ("eager", "flash_attention_2", "sdpa")

    Returns:
        Tuple of (model, tokenizer)
    """
    check_dependencies()

    print(f"\n📦 Loading base model: {model_name}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   4-bit quantization: {load_in_4bit}")
    print(f"   Attention: {attn_implementation}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,  # None = auto-detect (bfloat16 if supported, else float16)
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
        # Trust remote code for Qwen models
        trust_remote_code=True,
    )

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")

    print(f"   ✅ Model loaded successfully")

    return model, tokenizer


def setup_lora(
    model: Any,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = True,
    random_state: int = 42
) -> Any:
    """
    Apply LoRA adapters to model

    Args:
        model: Base model from load_base_model()
        lora_r: LoRA rank (higher = more capacity, 64 recommended)
        lora_alpha: LoRA alpha scaling (typically 2x rank)
        lora_dropout: LoRA dropout for regularization
        target_modules: Modules to apply LoRA (None = auto-detect optimal)
        use_gradient_checkpointing: Enable gradient checkpointing for memory
        random_state: Random seed for reproducibility

    Returns:
        Model with LoRA adapters applied
    """
    print(f"\n🔧 Applying LoRA adapters")
    print(f"   Rank (r): {lora_r}")
    print(f"   Alpha: {lora_alpha}")
    print(f"   Dropout: {lora_dropout}")

    # Default target modules for Qwen2.5 (covers attention + FFN + embeddings)
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # Feed-forward
            "embed_tokens", "lm_head"                  # Embeddings (CRITICAL for style)
        ]

    print(f"   Target modules: {', '.join(target_modules)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=False,  # Use standard LoRA
        loftq_config=None,
    )

    print(f"   ✅ LoRA applied successfully")

    return model


def load_training_data(
    train_path: str,
    val_path: Optional[str] = None,
    text_field: str = "text",
    max_samples: Optional[int] = None
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load training data from JSONL files

    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file (optional)
        text_field: Field name containing text (default "text")
        max_samples: Limit samples for testing (optional)

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print(f"\n📚 Loading training data")
    print(f"   Train: {train_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    train_dataset = load_dataset('json', data_files=train_path, split='train')

    if max_samples and len(train_dataset) > max_samples:
        print(f"   ⚠️  Limiting to {max_samples:,} training samples (test mode)")
        train_dataset = train_dataset.select(range(max_samples))

    print(f"   ✅ Train: {len(train_dataset):,} examples")

    eval_dataset = None
    if val_path:
        if not os.path.exists(val_path):
            print(f"   ⚠️  Validation data not found: {val_path}")
        else:
            eval_dataset = load_dataset('json', data_files=val_path, split='train')
            if max_samples and len(eval_dataset) > max_samples // 10:
                eval_dataset = eval_dataset.select(range(max_samples // 10))
            print(f"   ✅ Validation: {len(eval_dataset):,} examples")

    return train_dataset, eval_dataset


def formatting_func(examples: Dict[str, List]) -> List[str]:
    """
    Format examples for SFT training

    Converts ChatML-formatted examples to text strings.

    Args:
        examples: Batch of examples with 'messages' field

    Returns:
        List of formatted text strings
    """
    from itertools import chain

    texts = []
    for messages in examples.get('messages', []):
        if isinstance(messages, str):
            # Already formatted
            texts.append(messages)
        elif isinstance(messages, list):
            # Convert ChatML messages to text
            text_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            texts.append('\n'.join(text_parts))
        else:
            # Fallback
            texts.append(str(messages))

    return texts


def train_sft(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "checkpoints/sft",
    num_train_epochs: int = 5,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 16,
    warmup_ratio: float = 0.03,
    max_seq_length: int = 2048,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    optim: str = "paged_adamw_8bit",
    lr_scheduler_type: str = "cosine",
    seed: int = 42,
    report_to: str = "none",
    **kwargs
) -> Any:
    """
    Run supervised fine-tuning with QLoRA

    This is the primary training phase that learns the base personality.

    Args:
        model: Model with LoRA adapters from setup_lora()
        tokenizer: Tokenizer from load_base_model()
        train_dataset: Training dataset from load_training_data()
        eval_dataset: Optional validation dataset
        output_dir: Directory for checkpoints
        num_train_epochs: Number of training epochs (5 recommended)
        learning_rate: Learning rate (1e-4 recommended)
        per_device_train_batch_size: Batch size per GPU (2 for RTX 3070)
        gradient_accumulation_steps: Gradient accumulation (16 = effective batch 32)
        warmup_ratio: Warmup ratio (3% recommended)
        max_seq_length: Maximum sequence length
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        eval_steps: Evaluate every N steps
        max_grad_norm: Gradient clipping
        weight_decay: Weight decay for regularization
        optim: Optimizer (paged_adamw_8bit for memory efficiency)
        lr_scheduler_type: Learning rate scheduler
        seed: Random seed
        report_to: Logging backend ("none", "wandb", "tensorboard")
        **kwargs: Additional training arguments

    Returns:
        Trained model
    """
    print(f"\n🚀 Starting SFT Training")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {per_device_train_batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Prepare model for training
    model = FastLanguageModel.for_training(model)

    # Training arguments
    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        overwrite_output_dir=True,

        # Training
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,

        # Optimization
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        # Logging
        logging_steps=logging_steps,
        logging_dir=f"{output_dir}/logs",
        report_to=report_to,

        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,  # Keep only 3 most recent checkpoints

        # Evaluation
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=eval_steps if eval_dataset else None,
        per_device_eval_batch_size=per_device_train_batch_size,

        # Data
        remove_unused_columns=False,
        group_by_length=False,  # Keep natural length variation

        # Reproducibility
        seed=seed,
        data_seed=seed,

        # Hardware
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,

        # NO FILTERING (preserve authenticity)
        # These would be in dataset preprocessing, not here

        **kwargs
    )

    # Create SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,  # Don't pack multiple examples (preserve conversation structure)
    )

    # Train
    print("⏳ Training in progress...")
    print("   This will take 4-5 hours on RTX 3070 (5 epochs)")
    print("   Checkpoints saved every 500 steps\n")

    try:
        trainer.train()
        print(f"\n✅ SFT training complete!")
        print(f"   Final checkpoint: {output_dir}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory!")
            print(f"   Try reducing batch_size to 1 or gradient_accumulation_steps to 8")
            print(f"   Current: batch_size={per_device_train_batch_size}, grad_accum={gradient_accumulation_steps}")
            raise
        else:
            raise

    # Save final model
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print(f"   Saved final model: {output_dir}/final")

    return model


def load_dpo_data(
    dpo_path: str,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Load DPO preference pairs from JSONL

    Args:
        dpo_path: Path to DPO pairs JSONL file
        max_samples: Limit samples for testing (optional)

    Returns:
        Dataset with chosen/rejected pairs
    """
    print(f"\n💎 Loading DPO data")
    print(f"   Path: {dpo_path}")

    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"DPO data not found: {dpo_path}")

    dpo_dataset = load_dataset('json', data_files=dpo_path, split='train')

    if max_samples and len(dpo_dataset) > max_samples:
        print(f"   ⚠️  Limiting to {max_samples:,} pairs (test mode)")
        dpo_dataset = dpo_dataset.select(range(max_samples))

    print(f"   ✅ Loaded {len(dpo_dataset):,} preference pairs")

    return dpo_dataset


def train_dpo(
    model: Any,
    tokenizer: Any,
    dpo_dataset: Dataset,
    output_dir: str = "checkpoints/dpo",
    beta: float = 0.1,
    num_train_epochs: int = 2,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_seq_length: int = 2048,
    max_prompt_length: int = 1024,
    logging_steps: int = 10,
    save_steps: int = 500,
    seed: int = 42,
    report_to: str = "none",
    **kwargs
) -> Any:
    """
    Run Direct Preference Optimization on SFT model

    This reinforces preferred communication patterns using reaction data.

    Args:
        model: SFT-trained model from train_sft()
        tokenizer: Tokenizer
        dpo_dataset: DPO dataset with chosen/rejected pairs
        output_dir: Directory for checkpoints
        beta: DPO beta parameter (0.1 = moderate preference strength)
        num_train_epochs: Number of epochs (2 recommended)
        learning_rate: Learning rate (5e-5, lower than SFT)
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation
        max_seq_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        seed: Random seed
        report_to: Logging backend
        **kwargs: Additional DPO config arguments

    Returns:
        DPO-trained model
    """
    print(f"\n💎 Starting DPO Training")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Beta: {beta}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {per_device_train_batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Prepare model for DPO training
    model = FastLanguageModel.for_training(model)

    # DPO configuration
    dpo_config = DPOConfig(
        # Output
        output_dir=output_dir,
        overwrite_output_dir=True,

        # DPO-specific
        beta=beta,
        max_prompt_length=max_prompt_length,
        max_length=max_seq_length,

        # Training
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,

        # Optimization
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        # Logging
        logging_steps=logging_steps,
        logging_dir=f"{output_dir}/logs",
        report_to=report_to,

        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,

        # Reproducibility
        seed=seed,

        # Hardware
        remove_unused_columns=False,

        **kwargs
    )

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("⏳ DPO training in progress...")
    print("   This will take 1-2 hours on RTX 3070 (2 epochs)\n")

    try:
        trainer.train()
        print(f"\n✅ DPO training complete!")
        print(f"   Expected improvement: +20-30% style consistency")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory!")
            print(f"   Try reducing beta to 0.05 or batch_size to 1")
            raise
        else:
            raise

    # Save final model
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print(f"   Saved final model: {output_dir}/final")

    return model


def merge_and_save(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    save_method: str = "merged_16bit"
) -> str:
    """
    Merge LoRA weights into base model and save

    Args:
        model: Trained model with LoRA adapters
        tokenizer: Tokenizer
        output_dir: Output directory for merged model
        save_method: Save method ("merged_16bit", "merged_4bit", "lora")

    Returns:
        Path to saved model
    """
    print(f"\n💾 Merging and saving model")
    print(f"   Method: {save_method}")
    print(f"   Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    if save_method == "lora":
        # Save LoRA adapters only (small, need base model separately)
        model.save_pretrained(output_dir)
        print(f"   ✅ LoRA adapters saved")
    else:
        # Merge LoRA into base model
        print(f"   Merging LoRA weights...")
        model.save_pretrained_merged(
            output_dir,
            tokenizer,
            save_method=save_method
        )
        print(f"   ✅ Merged model saved")

    # Calculate size
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    size_gb = total_size / (1024 ** 3)
    print(f"   Model size: {size_gb:.2f} GB")

    return output_dir


if __name__ == "__main__":
    print("Model Trainer Module")
    print("=" * 60)
    print("\nThis module provides training functions for QLoRA + DPO.")
    print("\nKey functions:")
    print("  - load_base_model(): Load Qwen2.5-3B with 4-bit quantization")
    print("  - setup_lora(): Apply LoRA adapters")
    print("  - train_sft(): Supervised fine-tuning (4-5 hours)")
    print("  - train_dpo(): Direct preference optimization (1-2 hours)")
    print("  - merge_and_save(): Save final merged model")
    print("\nUsage:")
    print("  See scripts/3_train_model.py for orchestration")
    print("=" * 60)
