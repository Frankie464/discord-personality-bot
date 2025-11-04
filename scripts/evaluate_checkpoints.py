"""
Post-Training Checkpoint Evaluation Script

After training completes, use this script to evaluate all saved checkpoints
on the FULL validation set to determine which checkpoint is truly the best.

During training, we use a small validation subset (500 examples) for fast
evaluation (~2 minutes). After training, this script uses the full validation
set (36K+ examples) for comprehensive assessment (~1 hour per checkpoint).

Usage:
    # Evaluate all checkpoints in checkpoints/sft/
    python scripts/evaluate_checkpoints.py

    # Evaluate specific checkpoint directory
    python scripts/evaluate_checkpoints.py --checkpoint_dir checkpoints/sft

    # Use custom validation data
    python scripts/evaluate_checkpoints.py --val_data data_storage/training/val_sft_full.jsonl

    # Evaluate only specific checkpoint
    python scripts/evaluate_checkpoints.py --single checkpoints/sft/checkpoint-10000
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from model.trainer import load_base_model, setup_lora, load_training_data


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    Find all checkpoint directories in the given path

    Args:
        checkpoint_dir: Base checkpoint directory

    Returns:
        List of checkpoint paths sorted by step number
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return []

    # Find all checkpoint-* directories
    checkpoints = []
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith('checkpoint-'):
            checkpoints.append(str(item))

    # Sort by step number
    checkpoints.sort(key=lambda x: int(Path(x).name.split('-')[1]))

    return checkpoints


def evaluate_checkpoint(
    checkpoint_path: str,
    val_dataset: Any,
    tokenizer: Any
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate a single checkpoint on the validation set

    Args:
        checkpoint_path: Path to checkpoint directory
        val_dataset: Validation dataset
        tokenizer: Tokenizer

    Returns:
        Tuple of (eval_loss, metrics_dict)
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from model.trainer import FastLanguageModel, is_bfloat16_supported

    print(f"\n📊 Evaluating: {Path(checkpoint_path).name}")

    # Load model from checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Prepare for evaluation
    model = FastLanguageModel.for_inference(model)

    # Create minimal trainer just for evaluation
    training_args = TrainingArguments(
        output_dir="temp_eval",
        per_device_eval_batch_size=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # Formatting function (same as training)
    def formatting_func(example):
        if isinstance(example, dict):
            return [example.get('text', '')]
        return [example]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        max_seq_length=2048,
        args=training_args,
        packing=False,
    )

    # Run evaluation
    print(f"   Evaluating on {len(val_dataset)} examples...")
    start_time = datetime.now()

    metrics = trainer.evaluate()

    elapsed = (datetime.now() - start_time).total_seconds()
    eval_loss = metrics.get('eval_loss', float('inf'))

    print(f"   ✅ eval_loss: {eval_loss:.4f} (took {elapsed/60:.1f} minutes)")

    return eval_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate training checkpoints on full validation set')

    parser.add_argument(
        '--checkpoint_dir',
        default='checkpoints/sft',
        help='Directory containing checkpoints'
    )
    parser.add_argument(
        '--val_data',
        default='data_storage/training/val_sft_full.jsonl',
        help='Full validation data path'
    )
    parser.add_argument(
        '--single',
        default=None,
        help='Evaluate only a single checkpoint (path to checkpoint directory)'
    )
    parser.add_argument(
        '--output',
        default='checkpoints/evaluation_results.json',
        help='Output file for evaluation results'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📊 Checkpoint Evaluation on Full Validation Set")
    print("=" * 70)

    # Load validation data
    print(f"\n📂 Loading validation data...")
    print(f"   Path: {args.val_data}")

    if not os.path.exists(args.val_data):
        print(f"\n❌ Validation data not found: {args.val_data}")
        print(f"   Make sure you've run: python scripts/2_prepare_training_data.py")
        return 1

    _, val_dataset = load_training_data(
        train_path=None,  # Don't need training data
        val_path=args.val_data,
        max_samples=None
    )

    print(f"   ✅ Loaded {len(val_dataset)} validation examples")

    # Find checkpoints
    if args.single:
        if not os.path.exists(args.single):
            print(f"\n❌ Checkpoint not found: {args.single}")
            return 1
        checkpoints = [args.single]
        print(f"\n📁 Evaluating single checkpoint: {args.single}")
    else:
        checkpoints = find_checkpoints(args.checkpoint_dir)
        if not checkpoints:
            print(f"\n❌ No checkpoints found in: {args.checkpoint_dir}")
            return 1
        print(f"\n📁 Found {len(checkpoints)} checkpoints:")
        for cp in checkpoints:
            print(f"   - {Path(cp).name}")

    # Load tokenizer (same for all checkpoints)
    print(f"\n🔧 Loading tokenizer...")
    _, tokenizer = load_base_model(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True
    )

    # Evaluate each checkpoint
    results = []
    best_checkpoint = None
    best_loss = float('inf')

    print(f"\n" + "=" * 70)
    print(f"Starting evaluation (this will take ~{len(checkpoints) * 1} hour)")
    print(f"=" * 70)

    for checkpoint in checkpoints:
        try:
            eval_loss, metrics = evaluate_checkpoint(
                checkpoint,
                val_dataset,
                tokenizer
            )

            result = {
                'checkpoint': checkpoint,
                'checkpoint_name': Path(checkpoint).name,
                'eval_loss': eval_loss,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

            # Track best
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_checkpoint = checkpoint
                print(f"   ✨ NEW BEST MODEL!")

        except Exception as e:
            print(f"   ❌ Evaluation failed: {str(e)}")
            continue

    # Save results
    print(f"\n💾 Saving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'val_data': args.val_data,
        'num_val_examples': len(val_dataset),
        'num_checkpoints_evaluated': len(results),
        'best_checkpoint': best_checkpoint,
        'best_eval_loss': best_loss,
        'results': results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n" + "=" * 70)
    print(f"📊 EVALUATION SUMMARY")
    print(f"=" * 70)
    print(f"\nEvaluated {len(results)} checkpoints on {len(val_dataset)} validation examples\n")

    # Sort by eval_loss
    results_sorted = sorted(results, key=lambda x: x['eval_loss'])

    print("Rank | Checkpoint          | Eval Loss")
    print("-----+---------------------+-----------")
    for rank, result in enumerate(results_sorted, 1):
        marker = "✨" if result['checkpoint'] == best_checkpoint else "  "
        print(f" {rank:2d}  | {result['checkpoint_name']:19s} | {result['eval_loss']:.4f} {marker}")

    print(f"\n✨ BEST CHECKPOINT: {Path(best_checkpoint).name}")
    print(f"   Eval Loss: {best_loss:.4f}")
    print(f"\n📁 Use this checkpoint for final model:")
    print(f"   {best_checkpoint}")

    print(f"\n✅ Complete! Results saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
