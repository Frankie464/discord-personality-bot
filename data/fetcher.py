"""
Discord Message Fetcher - Incremental Ingestion with Channel Allowlist

This module implements incremental message fetching for the v2.0 architecture:
- Fetches ONLY new messages since last_message_id (not full history)
- Respects channel allowlist from database (enabled=1 only)
- Stores messages in SQLite database
- Updates last_fetch_message_id after successful fetch
- SILENT operation (no announcements)

Process Split (v2.0):
- This module is for DATA COLLECTION only
- Runs separately from bot (via scripts/fetch_and_embed.py)
- Scheduled manually or via cron/Task Scheduler
- Bot does NOT fetch messages (runs 24/7 separately)

Key Changes from v1.0:
- Incremental ingestion (not full scraping)
- Channel allowlist integration
- Database storage (not just JSON)
- Removed PrivacyManager (simplified to basic filtering)
"""

import discord
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from storage.database import Database
from data.privacy import should_include_message


class IncrementalMessageFetcher:
    """
    Fetches new Discord messages incrementally for training data

    Features:
    - Channel allowlist enforcement (only enabled=1 channels)
    - Incremental fetching (since last_message_id)
    - Database storage with deduplication
    - SILENT operation (no announcements)
    - Rate limiting (respect Discord API)
    """

    def __init__(
        self,
        bot_token: str,
        database: Database,
        output_dir: str = "data_storage/messages"
    ):
        """
        Initialize incremental message fetcher

        Args:
            bot_token: Discord bot token
            database: Database instance for allowlist and storage
            output_dir: Directory for JSON backups (optional)
        """
        self.bot_token = bot_token
        self.db = database
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True

        self.client = discord.Client(intents=intents)

        # Statistics
        self.stats = {
            'channels_processed': 0,
            'total_fetched': 0,
            'total_new': 0,
            'total_duplicates': 0,
            'total_filtered': 0,
            'by_channel': {},
            'errors': []
        }

    async def connect(self):
        """Connect to Discord"""
        await self.client.login(self.bot_token)

    async def close(self):
        """Close Discord connection"""
        await self.client.close()

    def _message_to_dict(self, message: discord.Message) -> Dict[str, Any]:
        """
        Convert Discord message to dictionary

        Args:
            message: Discord message object

        Returns:
            Dictionary with message data
        """
        # Count total reactions
        total_reactions = sum(reaction.count for reaction in message.reactions)

        # Extract reaction details
        reactions = []
        for reaction in message.reactions:
            reactions.append({
                'emoji': str(reaction.emoji),
                'count': reaction.count
            })

        # Extract reply_to if it's a reply
        reply_to = None
        if message.reference and message.reference.message_id:
            reply_to = str(message.reference.message_id)

        return {
            'message_id': str(message.id),
            'author_id': str(message.author.id),
            'username': str(message.author),
            'content': message.content,
            'timestamp': message.created_at.isoformat(),
            'channel_id': str(message.channel.id),
            'channel_name': message.channel.name if hasattr(message.channel, 'name') else 'Unknown',
            'reactions': total_reactions,  # Total count for balancing
            'reaction_details': reactions,  # Detailed breakdown
            'reply_to': reply_to,
            'is_bot': message.author.bot,
            'type': message.type.value,
            'has_attachments': len(message.attachments) > 0
        }

    async def fetch_channel_incremental(
        self,
        channel_id: str,
        last_message_id: Optional[str] = None,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch new messages from channel since last_message_id

        Args:
            channel_id: Discord channel ID
            last_message_id: Last fetched message ID (None = fetch all)
            show_progress: Show progress bar

        Returns:
            List of new message dictionaries
        """
        try:
            channel = await self.client.fetch_channel(int(channel_id))
        except discord.errors.Forbidden:
            error = f"No access to channel {channel_id}"
            self.stats['errors'].append(error)
            print(f"❌ {error}")
            return []
        except discord.errors.NotFound:
            error = f"Channel {channel_id} not found"
            self.stats['errors'].append(error)
            print(f"❌ {error}")
            return []
        except Exception as e:
            error = f"Error accessing channel {channel_id}: {e}"
            self.stats['errors'].append(error)
            print(f"❌ {error}")
            return []

        messages = []
        fetched_count = 0
        filtered_count = 0

        # Progress bar
        pbar = None
        if show_progress:
            pbar = tqdm(
                desc=f"Fetching {channel.name}",
                unit=" msgs",
                dynamic_ncols=True
            )

        try:
            # Determine starting point
            # If last_message_id provided, fetch only newer messages
            after = None
            if last_message_id:
                after = discord.Object(id=int(last_message_id))

            # Fetch messages
            async for message in channel.history(
                limit=None,  # Get all new messages
                after=after,  # Only messages after this ID
                oldest_first=False  # Newest first
            ):
                fetched_count += 1
                message_dict = self._message_to_dict(message)

                # Apply minimal filtering
                if should_include_message(message_dict):
                    messages.append(message_dict)
                else:
                    filtered_count += 1

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'new': len(messages),
                        'filtered': filtered_count
                    })

                # Rate limiting
                if fetched_count % 100 == 0:
                    await asyncio.sleep(0.5)

        except discord.errors.Forbidden:
            error = f"Lost access to channel {channel.name}"
            self.stats['errors'].append(error)
            print(f"❌ {error}")
        except Exception as e:
            error = f"Error fetching from {channel.name}: {e}"
            self.stats['errors'].append(error)
            print(f"❌ {error}")
        finally:
            if pbar:
                pbar.close()

        # Update statistics
        self.stats['by_channel'][channel_id] = {
            'name': channel.name if hasattr(channel, 'name') else 'Unknown',
            'fetched': fetched_count,
            'new': len(messages),
            'filtered': filtered_count
        }

        self.stats['total_fetched'] += fetched_count
        self.stats['total_filtered'] += filtered_count

        return messages

    def _store_messages_in_database(
        self,
        messages: List[Dict[str, Any]],
        channel_id: str
    ) -> int:
        """
        Store messages in SQLite database with deduplication

        Args:
            messages: List of message dictionaries
            channel_id: Channel ID

        Returns:
            Number of new messages added (duplicates skipped)
        """
        new_count = 0

        for msg in messages:
            # Check if message already exists
            existing = self.db.get_message_by_id(msg['message_id'])
            if existing:
                self.stats['total_duplicates'] += 1
                continue

            # Add to database
            self.db.add_message(
                message_id=msg['message_id'],
                author_id=msg['author_id'],
                content=msg['content'],
                timestamp=datetime.fromisoformat(msg['timestamp']),
                channel_id=msg['channel_id'],
                reactions=msg['reactions'],
                metadata=msg  # Store full message dict as metadata
            )
            new_count += 1

        return new_count

    def _save_json_backup(
        self,
        messages: List[Dict[str, Any]],
        channel_id: str
    ):
        """
        Save messages to JSON backup file (optional)

        Args:
            messages: List of message dictionaries
            channel_id: Channel ID for filename
        """
        if not messages:
            return

        filename = os.path.join(
            self.output_dir,
            f"channel_{channel_id}_incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

        print(f"💾 JSON backup: {filename}")

    async def fetch_from_allowlist(
        self,
        save_json_backup: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch new messages from all allowlisted channels

        This is the main entry point for incremental ingestion.

        Args:
            save_json_backup: Save JSON backups in addition to database

        Returns:
            Statistics dictionary
        """
        print("\n🔄 Starting Incremental Message Ingestion")
        print("=" * 60)

        # Get allowlisted channels from database
        allowed_channels = self.db.get_allowed_channels(enabled_only=True)

        if not allowed_channels:
            print("⚠️  No channels in allowlist. Add channels to database first.")
            return self.get_statistics()

        print(f"📋 Allowlisted channels: {len(allowed_channels)}")
        print("")

        for channel in allowed_channels:
            channel_id = channel['channel_id']
            channel_name = channel['channel_name']
            last_message_id = channel.get('last_fetch_message_id')

            print(f"📡 Processing: {channel_name} ({channel_id})")
            if last_message_id:
                print(f"   Fetching since message ID: {last_message_id}")
            else:
                print(f"   First fetch (no last_message_id)")

            # Fetch new messages
            messages = await self.fetch_channel_incremental(
                channel_id=channel_id,
                last_message_id=last_message_id,
                show_progress=True
            )

            if not messages:
                print(f"   No new messages")
                continue

            # Store in database
            new_count = self._store_messages_in_database(messages, channel_id)
            self.stats['total_new'] += new_count

            print(f"   💾 Stored: {new_count} new messages")
            print(f"   ⏭  Skipped: {len(messages) - new_count} duplicates")

            # Update last_fetch_message_id in allowlist
            if messages:
                # Get newest message ID (messages are newest-first)
                newest_message_id = messages[0]['message_id']
                self.db.update_channel_last_fetch(channel_id, newest_message_id)
                print(f"   ✅ Updated last_fetch_message_id: {newest_message_id}")

            # Optional JSON backup
            if save_json_backup:
                self._save_json_backup(messages, channel_id)

            self.stats['channels_processed'] += 1

            # Brief pause between channels
            await asyncio.sleep(1.0)
            print("")

        return self.get_statistics()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get fetch statistics

        Returns:
            Dictionary with statistics
        """
        return {
            'channels_processed': self.stats['channels_processed'],
            'total_fetched': self.stats['total_fetched'],
            'total_new': self.stats['total_new'],
            'total_duplicates': self.stats['total_duplicates'],
            'total_filtered': self.stats['total_filtered'],
            'by_channel': self.stats['by_channel'],
            'errors': self.stats['errors']
        }

    def print_statistics(self):
        """Print detailed statistics"""
        stats = self.get_statistics()

        print("=" * 60)
        print("📊 INCREMENTAL INGESTION STATISTICS")
        print("=" * 60)
        print(f"Channels Processed:      {stats['channels_processed']}")
        print(f"Total Messages Fetched:  {stats['total_fetched']:,}")
        print(f"New Messages Stored:     {stats['total_new']:,}")
        print(f"Duplicates Skipped:      {stats['total_duplicates']:,}")
        print(f"Filtered Out:            {stats['total_filtered']:,}")

        if stats['by_channel']:
            print("\nBy Channel:")
            for channel_id, channel_stats in stats['by_channel'].items():
                print(f"  {channel_stats['name']}:")
                print(f"    Fetched: {channel_stats['fetched']:,}")
                print(f"    New: {channel_stats['new']:,}")
                print(f"    Filtered: {channel_stats['filtered']:,}")

        if stats['errors']:
            print(f"\n⚠️  Errors encountered: {len(stats['errors'])}")
            for error in stats['errors']:
                print(f"  - {error}")

        print("=" * 60)


async def fetch_incremental_async(
    bot_token: str,
    database: Database,
    save_json_backup: bool = False
) -> Dict[str, Any]:
    """
    Async wrapper for incremental message fetching

    Args:
        bot_token: Discord bot token
        database: Database instance
        save_json_backup: Save JSON backups

    Returns:
        Statistics dictionary
    """
    fetcher = IncrementalMessageFetcher(bot_token, database)

    try:
        # Connect to Discord
        await fetcher.connect()

        # Fetch from allowlist
        stats = await fetcher.fetch_from_allowlist(save_json_backup)

        # Print statistics
        fetcher.print_statistics()

        return stats

    finally:
        # Always close connection
        await fetcher.close()


if __name__ == "__main__":
    # Test incremental fetcher
    print("Incremental Message Fetcher Module")
    print("=" * 60)
    print("\nv2.0 Architecture - Incremental Ingestion")
    print("\nKey Features:")
    print("  ✅ Fetches ONLY new messages since last_message_id")
    print("  ✅ Respects channel allowlist (enabled=1 only)")
    print("  ✅ Stores in SQLite database with deduplication")
    print("  ✅ Updates last_fetch_message_id after fetch")
    print("  ✅ SILENT operation (no announcements)")
    print("  ✅ Minimal filtering (bots, system messages only)")
    print("\nProcess Split:")
    print("  - This runs SEPARATELY from bot")
    print("  - Scheduled via cron/Task Scheduler")
    print("  - Bot does NOT fetch messages (runs 24/7 separately)")
    print("\nUsage:")
    print("  See scripts/fetch_and_embed.py for usage example")
    print("  Run manually or schedule weekly for incremental updates")
    print("=" * 60)
