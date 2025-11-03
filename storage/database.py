"""
Database module for Discord Personality Bot

Manages SQLite database for:
- Bot configuration (response rate, temperature, etc.)
- Response statistics
- Admin-only user exclusions (privacy controls - legacy)
- Conversation context tracking
- Channel allowlist (training data transparency)

Privacy Note: Designed for private servers (~30 people, trusted friends).
No complex opt-out systems. Basic filtering via channel allowlist.
"""

import sqlite3
import os
from datetime import datetime
from typing import Any, Optional, List, Dict
from contextlib import contextmanager


class Database:
    """SQLite database manager for bot state and configuration"""

    def __init__(self, db_path: str):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database schema
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_schema(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Configuration table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    messages_seen INTEGER DEFAULT 0,
                    responses_sent INTEGER DEFAULT 0,
                    avg_response_time REAL DEFAULT 0.0,
                    errors INTEGER DEFAULT 0
                )
            """)

            # Admin-only user exclusions (CRITICAL for privacy)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS excluded_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT,
                    excluded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT,
                    excluded_by_admin TEXT
                )
            """)

            # Create index on excluded_users for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_excluded_users_id
                ON excluded_users(user_id)
            """)

            # Conversation context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_context (
                    channel_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    username TEXT,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (channel_id, message_id)
                )
            """)

            # Create index for faster context queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_channel_time
                ON conversation_context(channel_id, timestamp DESC)
            """)

            # Channel allowlist table (NEW - for training data transparency)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS channel_allowlist (
                    channel_id TEXT PRIMARY KEY,
                    channel_name TEXT,
                    enabled INTEGER DEFAULT 1,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_fetch_message_id TEXT,
                    last_fetch_at TIMESTAMP
                )
            """)

            # Create index for enabled channels
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel_allowlist_enabled
                ON channel_allowlist(enabled)
            """)

            # Messages table (for training data deduplication)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    reactions TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for message queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_channel
                ON messages(channel_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_author
                ON messages(author_id)
            """)

            # Insert default configuration if not exists
            self._insert_default_config(cursor)

    def _insert_default_config(self, cursor):
        """Insert default configuration values (v2.0 defaults for private servers)"""
        defaults = {
            'response_rate': '0.05',  # 5%
            'temperature': '0.7',  # Updated for v2.0
            'top_p': '0.9',
            'top_k': '40',
            'max_tokens': '120',  # Updated for v2.0 (typical Discord message length)
            'repetition_penalty': '1.1',
            'model_context_length': '2048',
            'model_threads': '0',  # Auto-detect
            'model_chat_template': 'chatml',  # NEW: CRITICAL for Qwen2.5
            'gpu_layers': '0',  # NEW: GPU offloading (0=CPU only)
            'always_respond_to_mentions': 'true',
            'respond_only_to_mentions': 'false',  # NEW: Override response_rate
        }

        for key, value in defaults.items():
            cursor.execute("""
                INSERT OR IGNORE INTO config (key, value)
                VALUES (?, ?)
            """, (key, value))

    # ===== Configuration Methods =====

    def get_config(self, key: str) -> Optional[str]:
        """
        Get configuration value

        Args:
            key: Configuration key

        Returns:
            Configuration value or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row['value'] if row else None

    def set_config(self, key: str, value: str):
        """
        Set configuration value

        Args:
            key: Configuration key
            value: Configuration value
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO config (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, value, datetime.now()))

    def get_all_config(self) -> Dict[str, str]:
        """
        Get all configuration as dictionary

        Returns:
            Dictionary of all configuration key-value pairs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM config")
            return {row['key']: row['value'] for row in cursor.fetchall()}

    # ===== Admin-Only Exclusion Methods =====

    def add_excluded_user(
        self,
        user_id: str,
        username: str,
        reason: str,
        excluded_by_admin: str
    ):
        """
        Admin-only: Add user to exclusion list

        Args:
            user_id: Discord user ID to exclude
            username: Discord username
            reason: Reason for exclusion
            excluded_by_admin: Admin user ID who made the exclusion
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO excluded_users
                (user_id, username, excluded_at, reason, excluded_by_admin)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, username, datetime.now(), reason, excluded_by_admin))

    def is_user_excluded(self, user_id: str) -> bool:
        """
        Check if user is excluded by admin

        Args:
            user_id: Discord user ID to check

        Returns:
            True if user is excluded, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM excluded_users WHERE user_id = ?",
                (user_id,)
            )
            return cursor.fetchone() is not None

    def get_excluded_users(self) -> List[Dict[str, Any]]:
        """
        Admin-only: Get all excluded users

        Returns:
            List of dictionaries with exclusion details
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, excluded_at, reason, excluded_by_admin
                FROM excluded_users
                ORDER BY excluded_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def remove_excluded_user(self, user_id: str):
        """
        Admin-only: Remove user from exclusion list

        Args:
            user_id: Discord user ID to remove from exclusions
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM excluded_users WHERE user_id = ?",
                (user_id,)
            )

    def get_excluded_user_ids(self) -> List[str]:
        """
        Get list of all excluded user IDs (for filtering)

        Returns:
            List of excluded Discord user IDs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM excluded_users")
            return [row['user_id'] for row in cursor.fetchall()]

    # ===== Statistics Methods =====

    def log_statistics(
        self,
        messages_seen: int = 0,
        responses_sent: int = 0,
        avg_response_time: float = 0.0,
        errors: int = 0
    ):
        """
        Log bot statistics

        Args:
            messages_seen: Number of messages seen
            responses_sent: Number of responses sent
            avg_response_time: Average response time in seconds
            errors: Number of errors encountered
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO statistics
                (messages_seen, responses_sent, avg_response_time, errors)
                VALUES (?, ?, ?, ?)
            """, (messages_seen, responses_sent, avg_response_time, errors))

    def get_statistics(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get statistics for the last N hours

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with aggregated statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    SUM(messages_seen) as total_messages_seen,
                    SUM(responses_sent) as total_responses_sent,
                    AVG(avg_response_time) as avg_response_time,
                    SUM(errors) as total_errors,
                    COUNT(*) as data_points
                FROM statistics
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """, (hours,))

            row = cursor.fetchone()
            if row and row['data_points'] > 0:
                return {
                    'messages_seen': row['total_messages_seen'] or 0,
                    'responses_sent': row['total_responses_sent'] or 0,
                    'avg_response_time': row['avg_response_time'] or 0.0,
                    'errors': row['total_errors'] or 0,
                    'response_rate': (
                        (row['total_responses_sent'] / row['total_messages_seen'] * 100)
                        if row['total_messages_seen'] > 0 else 0.0
                    )
                }
            else:
                return {
                    'messages_seen': 0,
                    'responses_sent': 0,
                    'avg_response_time': 0.0,
                    'errors': 0,
                    'response_rate': 0.0
                }

    # ===== Training Message Methods =====

    def add_message(
        self,
        message_id: str,
        channel_id: str,
        author_id: str,
        content: str,
        timestamp,
        reactions: Optional[Any] = None,
        metadata: Optional[Any] = None
    ):
        """
        Add a message to the messages table for training data deduplication

        Args:
            message_id: Discord message ID
            channel_id: Discord channel ID
            author_id: Discord author ID
            content: Message content
            timestamp: Message timestamp (datetime object or ISO string)
            reactions: Reaction data (will be JSON serialized)
            metadata: Additional metadata (will be JSON serialized)
        """
        import json
        from datetime import datetime

        # Convert timestamp to string if it's a datetime object
        if isinstance(timestamp, datetime):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp)

        # Serialize reactions and metadata to JSON
        reactions_json = json.dumps(reactions) if reactions is not None else None
        metadata_json = json.dumps(metadata) if metadata is not None else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO messages
                (message_id, channel_id, author_id, content, timestamp, reactions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (message_id, channel_id, author_id, content, timestamp_str, reactions_json, metadata_json))

    def get_message_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a message by ID from the messages table

        Args:
            message_id: Discord message ID

        Returns:
            Message dict if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_id, channel_id, author_id, content, timestamp
                FROM messages
                WHERE message_id = ?
            """, (message_id,))

            row = cursor.fetchone()
            if row:
                return {
                    'message_id': row['message_id'],
                    'channel_id': row['channel_id'],
                    'author_id': row['author_id'],
                    'content': row['content'],
                    'timestamp': row['timestamp']
                }
            return None

    # ===== Conversation Context Methods =====

    def add_conversation_message(
        self,
        channel_id: str,
        message_id: str,
        user_id: str,
        username: str,
        content: str
    ):
        """
        Add message to conversation context

        Args:
            channel_id: Discord channel ID
            message_id: Discord message ID
            user_id: Discord user ID
            username: Discord username
            content: Message content
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO conversation_context
                (channel_id, message_id, user_id, username, content, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (channel_id, message_id, user_id, username, content, datetime.now()))

    def get_conversation_context(
        self,
        channel_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation context for a channel

        Args:
            channel_id: Discord channel ID
            limit: Maximum number of messages to retrieve

        Returns:
            List of recent messages (newest first)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_id, user_id, username, content, timestamp
                FROM conversation_context
                WHERE channel_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (channel_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def clear_old_conversation_context(self, days: int = 7):
        """
        Clear conversation context older than N days

        Args:
            days: Number of days to keep
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM conversation_context
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))

    # ===== Channel Allowlist Methods (NEW - Training Data Transparency) =====

    def add_channel_to_allowlist(
        self,
        channel_id: str,
        channel_name: str,
        enabled: bool = True
    ):
        """
        Add channel to allowlist (contributes to training data)

        Args:
            channel_id: Discord channel ID
            channel_name: Discord channel name
            enabled: Whether channel is enabled for data collection
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO channel_allowlist
                (channel_id, channel_name, enabled, added_at)
                VALUES (?, ?, ?, ?)
            """, (channel_id, channel_name, 1 if enabled else 0, datetime.now()))

    def remove_channel_from_allowlist(self, channel_id: str):
        """
        Remove channel from allowlist

        Args:
            channel_id: Discord channel ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM channel_allowlist WHERE channel_id = ?",
                (channel_id,)
            )

    def is_channel_allowed(self, channel_id: str) -> bool:
        """
        Check if channel is in allowlist and enabled

        Args:
            channel_id: Discord channel ID

        Returns:
            True if channel is allowed and enabled, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT enabled FROM channel_allowlist WHERE channel_id = ?",
                (channel_id,)
            )
            row = cursor.fetchone()
            return row is not None and row['enabled'] == 1

    def get_allowed_channels(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all channels in allowlist

        Args:
            enabled_only: If True, only return enabled channels

        Returns:
            List of dictionaries with channel details
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if enabled_only:
                cursor.execute("""
                    SELECT channel_id, channel_name, enabled, added_at,
                           last_fetch_message_id, last_fetch_at
                    FROM channel_allowlist
                    WHERE enabled = 1
                    ORDER BY channel_name
                """)
            else:
                cursor.execute("""
                    SELECT channel_id, channel_name, enabled, added_at,
                           last_fetch_message_id, last_fetch_at
                    FROM channel_allowlist
                    ORDER BY channel_name
                """)
            return [dict(row) for row in cursor.fetchall()]

    def update_channel_last_fetch(
        self,
        channel_id: str,
        last_message_id: str
    ):
        """
        Update last fetch information for a channel

        Args:
            channel_id: Discord channel ID
            last_message_id: ID of last message fetched
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE channel_allowlist
                SET last_fetch_message_id = ?,
                    last_fetch_at = ?
                WHERE channel_id = ?
            """, (last_message_id, datetime.now(), channel_id))

    def enable_channel(self, channel_id: str):
        """
        Enable channel in allowlist

        Args:
            channel_id: Discord channel ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE channel_allowlist
                SET enabled = 1
                WHERE channel_id = ?
            """, (channel_id,))

    def disable_channel(self, channel_id: str):
        """
        Disable channel in allowlist (keeps in database but stops data collection)

        Args:
            channel_id: Discord channel ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE channel_allowlist
                SET enabled = 0
                WHERE channel_id = ?
            """, (channel_id,))


# Convenience functions for module-level usage

def init_database(db_path: str = "data_storage/database/bot.db") -> Database:
    """
    Initialize database with default path

    Args:
        db_path: Path to database file

    Returns:
        Database instance
    """
    return Database(db_path)


if __name__ == "__main__":
    # Test database creation
    print("Testing database initialization...")
    db = init_database("data_storage/database/bot.db")
    print("✅ Database initialized successfully!")

    # Test configuration (v2.0 defaults)
    print("\nTesting configuration...")
    print(f"Response rate: {db.get_config('response_rate')}")
    print(f"Temperature: {db.get_config('temperature')} (should be 0.7)")
    print(f"Max tokens: {db.get_config('max_tokens')} (should be 120)")
    print(f"Chat template: {db.get_config('model_chat_template')} (should be chatml)")
    print(f"GPU layers: {db.get_config('gpu_layers')} (should be 0)")

    # Test channel allowlist (NEW)
    print("\nTesting channel allowlist...")
    db.add_channel_to_allowlist(
        channel_id="111222333",
        channel_name="general",
        enabled=True
    )
    db.add_channel_to_allowlist(
        channel_id="444555666",
        channel_name="chat",
        enabled=True
    )
    print(f"Channel allowed: {db.is_channel_allowed('111222333')}")
    allowed_channels = db.get_allowed_channels()
    print(f"Allowed channels: {len(allowed_channels)}")
    for channel in allowed_channels:
        print(f"  - {channel['channel_name']} (ID: {channel['channel_id']})")

    # Test exclusion system (legacy)
    print("\nTesting exclusion system (legacy)...")
    db.add_excluded_user(
        user_id="123456789",
        username="test_user",
        reason="Testing",
        excluded_by_admin="admin_id"
    )
    print(f"User excluded: {db.is_user_excluded('123456789')}")
    print(f"Excluded users: {len(db.get_excluded_users())}")

    print("\n✅ All tests passed!")
