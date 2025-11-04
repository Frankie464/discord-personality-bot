"""
Training Data Preparation Script

Prepares Discord messages for QLoRA fine-tuning:
- Loads messages from SQLite database
- Applies user balancing (prevents single-user dominance)
- Filters training data (minimal filtering for authenticity)
- Formats as ChatML for Qwen2.5-3B-Instruct
- Creates DPO preference pairs from reactions
- Splits into train/validation/test sets
- Saves as JSONL files for training

This script must be run BEFORE model training (scripts/3_train_model.py).

Usage:
    # Full dataset
    python scripts/2_prepare_training_data.py

    # Test with limited messages
    python scripts/2_prepare_training_data.py --limit 1000

    # Custom output directory
    python scripts/2_prepare_training_data.py --output data_storage/training_custom/

    # Custom train/val/test split
    python scripts/2_prepare_training_data.py --train_ratio 0.8 --val_ratio 0.15 --test_ratio 0.05
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from storage.database import Database
from data.preprocessor import (
    filter_training_messages,
    calculate_user_weights,
    apply_balanced_sampling,
    format_for_training,
    create_dpo_pairs,
    get_balancing_statistics
)


def load_messages_from_database(
    db_path: str,
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Load messages from SQLite database

    Args:
        db_path: Path to SQLite database
        limit: Optional limit on number of messages to load (for testing)

    Returns:
        List of message dictionaries
    """
    print(f"\n📂 Loading messages from database...")
    print(f"   Database: {db_path}")

    db = Database(db_path)

    # Get all messages
    messages = db.get_all_messages()

    if limit and len(messages) > limit:
        print(f"   ⚠️  Limiting to {limit:,} messages (test mode)")
        # Random sample for representative test
        random.seed(42)
        messages = random.sample(messages, limit)

    print(f"   ✅ Loaded {len(messages):,} messages")

    return messages


def split_train_val_test(
    messages: List[Dict[str, Any]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split messages into train/validation/test sets

    Args:
        messages: List of message dictionaries
        train_ratio: Proportion for training (default 85%)
        val_ratio: Proportion for validation (default 10%)
        test_ratio: Proportion for test (default 5%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_messages, val_messages, test_messages)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
        "Ratios must sum to 1.0"

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    shuffled = messages.copy()
    random.shuffle(shuffled)

    # Calculate split indices
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_messages = shuffled[:train_end]
    val_messages = shuffled[train_end:val_end]
    test_messages = shuffled[val_end:]

    print(f"\n📊 Dataset Split:")
    print(f"   Train: {len(train_messages):,} messages ({len(train_messages)/total*100:.1f}%)")
    print(f"   Validation: {len(val_messages):,} messages ({len(val_messages)/total*100:.1f}%)")
    print(f"   Test: {len(test_messages):,} messages ({len(test_messages)/total*100:.1f}%)")

    return train_messages, val_messages, test_messages


def save_jsonl(examples: List[Dict[str, Any]], output_path: str):
    """
    Save training examples to JSONL format

    Args:
        examples: List of training examples
        output_path: Path to output JSONL file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')

    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   💾 {os.path.basename(output_path)}: {len(examples):,} examples ({file_size_mb:.1f} MB)")


def generate_statistics_report(
    original_messages: List[Dict],
    filtered_messages: List[Dict],
    train_examples: List[Dict],
    val_examples: List[Dict],
    test_examples: List[Dict],
    dpo_pairs: List[Dict],
    user_weights: Dict[str, float],
    output_dir: str
):
    """
    Generate comprehensive statistics report

    Args:
        original_messages: Original message list
        filtered_messages: After filtering
        train_examples: Training examples
        val_examples: Validation examples
        test_examples: Test examples
        dpo_pairs: DPO preference pairs
        user_weights: User weight dictionary
        output_dir: Directory to save report
    """
    stats = {
        'timestamp': datetime.now().isoformat(),
        'data_pipeline': {
            'original_messages': len(original_messages),
            'filtered_messages': len(filtered_messages),
            'filtered_out': len(original_messages) - len(filtered_messages),
            'filter_rate': f"{(len(original_messages) - len(filtered_messages)) / len(original_messages) * 100:.1f}%"
        },
        'dataset_split': {
            'train_examples': len(train_examples),
            'val_examples': len(val_examples),
            'test_examples': len(test_examples),
            'total_examples': len(train_examples) + len(val_examples) + len(test_examples)
        },
        'dpo_pairs': {
            'total_pairs': len(dpo_pairs)
        },
        'user_balancing': {
            'num_users': len(user_weights),
            'max_user_weight': max(user_weights.values()) if user_weights else 0,
            'min_user_weight': min(user_weights.values()) if user_weights else 0,
            'top_3_users': sorted(
                user_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
    }

    # Save JSON report
    report_path = os.path.join(output_dir, 'preparation_statistics.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 DATA PREPARATION STATISTICS")
    print(f"{'='*60}")
    print(f"\nOriginal Messages:      {stats['data_pipeline']['original_messages']:,}")
    print(f"After Filtering:        {stats['data_pipeline']['filtered_messages']:,}")
    print(f"Filtered Out:           {stats['data_pipeline']['filtered_out']:,} ({stats['data_pipeline']['filter_rate']})")
    print(f"\nTraining Examples:      {stats['dataset_split']['train_examples']:,}")
    print(f"Validation Examples:    {stats['dataset_split']['val_examples']:,}")
    print(f"Test Examples:          {stats['dataset_split']['test_examples']:,}")
    print(f"\nDPO Preference Pairs:   {stats['dpo_pairs']['total_pairs']:,}")
    print(f"\nUser Balancing:")
    print(f"  Total Users:          {stats['user_balancing']['num_users']}")
    print(f"  Max User Weight:      {stats['user_balancing']['max_user_weight']:.1%}")
    print(f"  Top 3 Contributors:")
    for user_id, weight in stats['user_balancing']['top_3_users']:
        print(f"    {user_id}: {weight:.1%}")
    print(f"\n✅ Report saved: {report_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Discord messages for model training'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='data_storage/database/bot.db',
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_storage/training/',
        help='Output directory for training files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of messages (for testing)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.85,
        help='Training set ratio (default 0.85)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.10,
        help='Validation set ratio (default 0.10)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.05,
        help='Test set ratio (default 0.05)'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default="Chat naturally, same tone as the server. Chaotic, unhinged blend of us. No rules. Do not talk like an AI or assistant.",
        help='System prompt for training (MUST match runtime prompt in model/prompts.py)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.database):
        print(f"❌ Error: Database not found: {args.database}")
        sys.exit(1)

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        print(f"❌ Error: Train/val/test ratios must sum to 1.0")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"🚀 TRAINING DATA PREPARATION")
    print(f"{'='*60}")
    print(f"Database: {args.database}")
    print(f"Output: {args.output}")
    if args.limit:
        print(f"⚠️  Test Mode: Limited to {args.limit:,} messages")
    print(f"{'='*60}")

    # Step 1: Load messages from database
    messages = load_messages_from_database(args.database, args.limit)

    if len(messages) < 100:
        print(f"\n⚠️  Warning: Only {len(messages)} messages found. Recommend at least 1,000 for quality training.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    original_count = len(messages)

    # Step 2: Filter messages (minimal filtering for authenticity)
    print(f"\n🔍 Filtering messages...")
    filtered_messages = filter_training_messages(messages)
    print(f"   ✅ Kept {len(filtered_messages):,} messages (filtered {original_count - len(filtered_messages):,})")

    # Step 3: Calculate user weights
    print(f"\n⚖️  Calculating user weights...")
    user_weights = calculate_user_weights(filtered_messages)
    balancing_stats = get_balancing_statistics(filtered_messages, user_weights)
    print(f"   ✅ Balanced {len(user_weights)} users")
    print(f"   Max user weight: {max(user_weights.values()):.1%}")

    # Step 4: Apply balanced sampling
    print(f"\n🎲 Applying balanced sampling...")
    balanced_messages = apply_balanced_sampling(
        filtered_messages,
        target_count=len(filtered_messages),  # Keep all, just reweight
        seed=args.seed
    )
    print(f"   ✅ Sampled {len(balanced_messages):,} messages")

    # Step 5: Split into train/val/test
    train_messages, val_messages, test_messages = split_train_val_test(
        balanced_messages,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    # Step 6: Format as ChatML for training
    print(f"\n📝 Formatting as ChatML...")
    print(f"   System prompt: \"{args.system_prompt}\"")

    train_examples = format_for_training(
        train_messages,
        system_prompt=args.system_prompt,
        context_window=5
    )
    print(f"   ✅ Train: {len(train_examples):,} conversation examples")

    val_examples = format_for_training(
        val_messages,
        system_prompt=args.system_prompt,
        context_window=5
    )
    print(f"   ✅ Validation (full): {len(val_examples):,} conversation examples")

    # Create small validation subset for fast training evaluation
    # Full validation set used for post-training assessment
    val_train_size = min(500, len(val_examples))  # 500 examples for training eval
    random.seed(args.seed)
    val_train_examples = random.sample(val_examples, val_train_size)
    print(f"   ✅ Validation (training): {len(val_train_examples):,} conversation examples (for fast eval)")

    test_examples = format_for_training(
        test_messages,
        system_prompt=args.system_prompt,
        context_window=5
    )
    print(f"   ✅ Test: {len(test_examples):,} conversation examples")

    # Step 7: Create DPO preference pairs
    print(f"\n💎 Creating DPO preference pairs...")
    dpo_pairs = create_dpo_pairs(
        balanced_messages,
        min_reactions=1,
        max_reactions_cap=5,  # Prevent outlier dominance
        min_length_tokens=4
    )
    print(f"   ✅ Created {len(dpo_pairs):,} preference pairs")

    # Step 8: Save all datasets
    print(f"\n💾 Saving training files...")
    os.makedirs(args.output, exist_ok=True)

    save_jsonl(train_examples, os.path.join(args.output, 'train_sft.jsonl'))
    save_jsonl(val_train_examples, os.path.join(args.output, 'val_sft_train.jsonl'))
    save_jsonl(val_examples, os.path.join(args.output, 'val_sft_full.jsonl'))
    save_jsonl(test_examples, os.path.join(args.output, 'test_sft.jsonl'))
    save_jsonl(dpo_pairs, os.path.join(args.output, 'dpo_pairs.jsonl'))

    # Step 9: Generate statistics report
    generate_statistics_report(
        messages,
        filtered_messages,
        train_examples,
        val_examples,
        test_examples,
        dpo_pairs,
        user_weights,
        args.output
    )

    print(f"\n✅ Data preparation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review statistics: {os.path.join(args.output, 'preparation_statistics.json')}")
    print(f"  2. Run training: python scripts/3_train_model.py")
    print(f"\nOutput files:")
    print(f"  - {os.path.join(args.output, 'train_sft.jsonl')}")
    print(f"  - {os.path.join(args.output, 'val_sft_train.jsonl')} ({len(val_train_examples)} examples - for training eval)")
    print(f"  - {os.path.join(args.output, 'val_sft_full.jsonl')} ({len(val_examples)} examples - for final eval)")
    print(f"  - {os.path.join(args.output, 'test_sft.jsonl')}")
    print(f"  - {os.path.join(args.output, 'dpo_pairs.jsonl')}")
    print(f"\nNote: Training uses val_sft_train.jsonl for fast evaluation (~2 min)")
    print(f"      Use scripts/evaluate_checkpoints.py after training for comprehensive eval")


if __name__ == '__main__':
    main()
