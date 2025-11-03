"""
Incremental Fetch and Embed Script

This script performs the complete incremental ingestion pipeline:
1. Fetch new messages from allowlisted channels (since last_message_id)
2. Store in SQLite database
3. Generate embeddings and store in LanceDB

v2.0 Architecture - Process Split:
- This script runs SEPARATELY from the 24/7 bot
- Scheduled manually or via cron/Task Scheduler
- Recommended: Run weekly for incremental updates
- Bot does NOT fetch messages (runs 24/7 independently)

Usage:
    python scripts/fetch_and_embed.py

Schedule (Windows Task Scheduler):
    schtasks /create /tn "BotDataFetch" /tr "C:\\path\\to\\venv\\Scripts\\python.exe C:\\path\\to\\scripts\\fetch_and_embed.py" /sc weekly /d SUN /st 03:00

Schedule (Linux/Mac cron):
    0 3 * * 0 /path/to/venv/bin/python /path/to/scripts/fetch_and_embed.py
"""

import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from storage.database import init_database
from storage.vectordb import VectorDatabase
from data.fetcher import fetch_incremental_async


def main():
    """Main entry point for incremental fetch and embed"""
    print("=" * 70)
    print("INCREMENTAL FETCH AND EMBED PIPELINE (v2.0)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load environment variables
    load_dotenv()

    # Get configuration
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    db_path = os.getenv('DATABASE_PATH', 'data_storage/database/bot.db')
    vector_db_path = os.getenv('VECTOR_DB_PATH', 'data_storage/embeddings')
    embedding_model = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')

    if not bot_token:
        print("❌ Error: DISCORD_BOT_TOKEN not found in .env file")
        print("   Please configure .env before running this script")
        sys.exit(1)

    print("📋 Configuration:")
    print(f"   Database: {db_path}")
    print(f"   Vector DB: {vector_db_path}")
    print(f"   Embedding Model: {embedding_model}")
    print()

    # Initialize database
    print("🗄️  Initializing database...")
    db = init_database(db_path)
    print("✅ Database ready")
    print()

    # Check channel allowlist
    allowed_channels = db.get_allowed_channels(enabled_only=True)
    if not allowed_channels:
        print("⚠️  WARNING: No channels in allowlist!")
        print("   ")
        print("   To add channels to the allowlist:")
        print("   1. Use bot command: !botdata")
        print("   2. Or manually add to database:")
        print("      >>> from storage.database import init_database")
        print("      >>> db = init_database('data_storage/database/bot.db')")
        print("      >>> db.add_channel_to_allowlist('CHANNEL_ID', 'channel-name')")
        print()
        sys.exit(1)

    print(f"✅ Found {len(allowed_channels)} allowlisted channel(s):")
    for ch in allowed_channels:
        print(f"   - {ch['channel_name']} ({ch['channel_id']})")
    print()

    # Step 1: Fetch messages incrementally
    print("=" * 70)
    print("STEP 1: INCREMENTAL MESSAGE FETCH")
    print("=" * 70)
    print()

    try:
        # Run async fetch
        stats = asyncio.run(fetch_incremental_async(
            bot_token=bot_token,
            database=db,
            save_json_backup=False  # Don't save JSON backups by default
        ))

        if stats['total_new'] == 0:
            print()
            print("✅ No new messages fetched (channels up to date)")
            print()
        else:
            print()
            print(f"✅ Fetched {stats['total_new']} new messages")
            print()

    except Exception as e:
        print()
        print(f"❌ Error during message fetch: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Generate embeddings
    print("=" * 70)
    print("STEP 2: GENERATE EMBEDDINGS")
    print("=" * 70)
    print()

    try:
        # Initialize vector database
        print(f"🧬 Initializing vector database...")
        print(f"   Model: {embedding_model}")

        vector_db = VectorDatabase(
            db_path=vector_db_path,
            embedding_model=embedding_model
        )
        print("✅ Vector database ready")
        print()

        # Get newly added messages (those without embeddings)
        print("📊 Fetching messages for embedding...")

        # Get ALL messages from database and check which need embeddings
        messages_to_embed = []

        for channel in allowed_channels:
            channel_id = channel['channel_id']

            # Get ALL messages from this channel (no limit)
            print(f"   Checking {channel['channel_name']} for messages without embeddings...")
            all_messages = db.get_messages_by_channel(
                channel_id=channel_id,
                limit=None  # Get ALL messages to check for embeddings
            )

            for msg in all_messages:
                # Check if already embedded
                if not vector_db.message_exists(msg['message_id']):
                    messages_to_embed.append(msg)

        if not messages_to_embed:
            print("✅ All messages already embedded")
            print()
        else:
            print(f"   Found {len(messages_to_embed)} messages to embed")
            print()

            # Add to vector database in batch
            print("🔄 Generating embeddings...")
            added_count = vector_db.add_messages_batch(messages_to_embed)

            print()
            print(f"✅ Added {added_count} embeddings to vector database")
            print()

        # Print vector DB stats
        vector_stats = vector_db.get_stats()
        print("📊 Vector Database Statistics:")
        print(f"   Total embeddings: {vector_stats['total_messages']:,}")
        print(f"   Table: {vector_stats['table_name']}")
        print(f"   Embedding dimensions: {vector_stats['embedding_model']}")
        print()

    except ImportError as e:
        print("⚠️  Warning: LanceDB or sentence-transformers not installed")
        print("   Skipping embedding generation")
        print("   Install with: pip install lancedb sentence-transformers")
        print()
    except Exception as e:
        print(f"⚠️  Warning: Error during embedding generation: {e}")
        print("   Messages were stored in database, but embeddings failed")
        print("   You can retry embedding later")
        print()
        import traceback
        traceback.print_exc()

    # Summary
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Summary:")
    print(f"  ✅ New messages fetched: {stats['total_new']}")
    print(f"  ✅ Messages embedded: {len(messages_to_embed) if 'messages_to_embed' in locals() else 0}")
    print(f"  ✅ Channels processed: {stats['channels_processed']}")
    print()

    if stats.get('errors'):
        print("⚠️  Errors encountered:")
        for error in stats['errors']:
            print(f"  - {error}")
        print()

    print("Next Steps:")
    print("  - Schedule this script to run weekly (recommended)")
    print("  - Check vector database stats: python -m storage.vectordb")
    print("  - Retrain model quarterly with new data: python scripts/3_train_model.py")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
