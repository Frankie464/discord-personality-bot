#!/usr/bin/env python3
"""
Script 1: Fetch All Message History

Scrapes Discord server message history for personality training.

Key Features:
- SILENT operation (no announcements to users)
- Respects admin exclusions only
- Fetches entire available history (typically 15-20 months)
- Target: 20,000-100,000+ messages
- Saves to JSON files per channel

Privacy:
- Admin-only exclusions
- No user-facing opt-out
- Server-wide personality blend

Usage:
    python scripts/1_fetch_all_history.py

    Or with custom limit for testing:
    python scripts/1_fetch_all_history.py --limit 100
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from data.fetcher import fetch_messages_async
from data.privacy import PrivacyManager
from storage.database import init_database


def load_configuration():
    """
    Load configuration from .env file

    Returns:
        Dictionary with configuration
    """
    # Load .env file
    env_path = project_root / ".env"
    if not env_path.exists():
        print("❌ Error: .env file not found!")
        print("   Please copy .env.example to .env and configure it.")
        sys.exit(1)

    load_dotenv(env_path)

    # Get required variables
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    server_id = os.getenv('DISCORD_SERVER_ID')
    channel_ids_str = os.getenv('DISCORD_CHANNEL_IDS')
    database_path = os.getenv('DATABASE_PATH', 'data_storage/database/bot.db')

    # Validate
    if not bot_token:
        print("❌ Error: DISCORD_BOT_TOKEN not set in .env")
        sys.exit(1)

    if not server_id:
        print("❌ Error: DISCORD_SERVER_ID not set in .env")
        sys.exit(1)

    if not channel_ids_str:
        print("❌ Error: DISCORD_CHANNEL_IDS not set in .env")
        sys.exit(1)

    # Parse channel IDs
    try:
        channel_ids = [int(cid.strip()) for cid in channel_ids_str.split(',')]
    except ValueError:
        print("❌ Error: Invalid DISCORD_CHANNEL_IDS format")
        print("   Expected: comma-separated list of channel IDs")
        sys.exit(1)

    return {
        'bot_token': bot_token,
        'server_id': int(server_id),
        'channel_ids': channel_ids,
        'database_path': database_path
    }


async def main():
    """Main function for message collection"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Discord message history for personality training"
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit messages per channel (default: unlimited)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_storage/messages',
        help='Output directory for messages (default: data_storage/messages)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🤖 Discord Personality Bot - Message History Fetcher")
    print("=" * 70)
    print("\nPHASE 1: Data Collection")
    print("\nPrivacy Model:")
    print("  • SILENT operation (no user announcements)")
    print("  • Admin-only exclusions")
    print("  • Server-wide personality blend")
    print("  • Minimal filtering (preserves authenticity)")
    print("")

    # Load configuration
    print("Loading configuration from .env...")
    config = load_configuration()
    print(f"✅ Bot token: {'*' * 8}{config['bot_token'][-4:]}")
    print(f"✅ Server ID: {config['server_id']}")
    print(f"✅ Channels to fetch: {len(config['channel_ids'])}")

    # Initialize database
    print("\nInitializing database...")
    db = init_database(config['database_path'])
    print(f"✅ Database initialized: {config['database_path']}")

    # Initialize privacy manager
    privacy_manager = PrivacyManager(db)
    excluded_count = privacy_manager.get_excluded_count()
    print(f"✅ Privacy manager ready (excluded users: {excluded_count})")

    if excluded_count > 0:
        print(f"\n⚠️  Admin has excluded {excluded_count} user(s) from training data")
        excluded_users = privacy_manager.get_excluded_users()
        for user in excluded_users:
            print(f"   - {user['username']} (ID: {user['user_id']})")
            print(f"     Reason: {user['reason']}")
            print(f"     By admin: {user['excluded_by_admin']}")

    # Confirm before starting
    print("\n" + "=" * 70)
    if args.limit:
        print(f"⚠️  TEST MODE: Limiting to {args.limit} messages per channel")
    else:
        print("📊 FULL COLLECTION MODE")
        print("   Target: 20,000-100,000+ messages")
        print("   This will fetch entire available history (~15-20 months)")
    print("=" * 70)

    input("\nPress Enter to start collection (or Ctrl+C to cancel)...")

    # Fetch messages
    print("\n🚀 Starting message collection...\n")

    try:
        stats = await fetch_messages_async(
            bot_token=config['bot_token'],
            server_id=config['server_id'],
            channel_ids=config['channel_ids'],
            privacy_manager=privacy_manager,
            output_dir=args.output,
            limit_per_channel=args.limit
        )

        # Summary
        print("\n" + "=" * 70)
        print("✅ COLLECTION COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   Total messages collected: {stats['total_included']:,}")
        print(f"   Unique users: {stats['unique_users']:,}")
        print(f"   Channels: {stats['channels']}")
        print(f"   Date range: {stats['date_range']['oldest']} to {stats['date_range']['newest']}")
        print(f"\n🔒 Privacy:")
        print(f"   Admin-excluded users: {stats['excluded_count']}")
        print(f"   Filter rate: {stats['filter_rate']:.1f}%")
        print(f"\n💾 Data saved to: {args.output}/")

        # Quality check
        print(f"\n✅ Quality Check:")
        if stats['total_included'] < 10000:
            print(f"   ⚠️  Low message count: {stats['total_included']:,}")
            print(f"      Target: 20,000+ for good personality capture")
            print(f"      Consider adding more channels or fetching longer history")
        elif stats['total_included'] < 20000:
            print(f"   ⚠️  Acceptable message count: {stats['total_included']:,}")
            print(f"      Target: 20,000+ recommended for best results")
        else:
            print(f"   ✅ Excellent message count: {stats['total_included']:,}")
            print(f"      Good data for personality training!")

        # Next steps
        print(f"\n📋 Next Steps:")
        print(f"   1. Review sample messages in {args.output}/")
        print(f"   2. Verify data quality (emojis, typos, slang preserved)")
        print(f"   3. Proceed to Phase 2: scripts/2_prepare_training_data.py")
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        print("\n\n❌ Collection cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
