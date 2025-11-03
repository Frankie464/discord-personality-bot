"""
Bot Configuration Module

Loads and validates configuration from environment variables.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv


class BotConfig:
    """Bot configuration loaded from .env file"""

    def __init__(self, env_path: str = ".env"):
        """
        Load configuration from .env file

        Args:
            env_path: Path to .env file
        """
        load_dotenv(env_path)

        # Discord Configuration
        self.bot_token = os.getenv('DISCORD_BOT_TOKEN')
        self.server_id = os.getenv('DISCORD_SERVER_ID')
        self.channel_ids_str = os.getenv('DISCORD_CHANNEL_IDS', '')
        self.admin_user_ids_str = os.getenv('ADMIN_USER_IDS', '')

        # Model Configuration (v2.0)
        self.model_path = os.getenv('MODEL_PATH', 'models/finetuned/qwen2.5-3b-personality-q4.gguf')
        self.model_chat_template = os.getenv('MODEL_CHAT_TEMPLATE', 'chatml')  # CRITICAL for Qwen2.5
        self.model_context_length = int(os.getenv('MODEL_CONTEXT_LENGTH', '2048'))
        self.model_threads = int(os.getenv('MODEL_THREADS', '0'))  # 0 = auto
        self.gpu_layers = int(os.getenv('GPU_LAYERS', '0'))  # Number of layers to offload to GPU

        # Generation Parameters (v2.0 defaults for private servers)
        self.temperature = float(os.getenv('GENERATION_TEMPERATURE', '0.7'))  # Updated from 0.75
        self.top_p = float(os.getenv('GENERATION_TOP_P', '0.9'))
        self.top_k = int(os.getenv('GENERATION_TOP_K', '40'))
        self.max_tokens = int(os.getenv('GENERATION_MAX_TOKENS', '120'))  # Updated from 150
        self.repetition_penalty = float(os.getenv('GENERATION_REPETITION_PENALTY', '1.1'))

        # Bot Behavior (v2.0)
        self.response_rate = float(os.getenv('RESPONSE_RATE', '0.05'))  # 5%
        self.always_respond_to_mentions = os.getenv('ALWAYS_RESPOND_TO_MENTIONS', 'true').lower() == 'true'
        self.respond_only_to_mentions = os.getenv('RESPOND_ONLY_TO_MENTIONS', 'false').lower() == 'true'

        # Database and Storage
        self.database_path = os.getenv('DATABASE_PATH', 'data_storage/database/bot.db')
        self.vector_db_path = os.getenv('VECTOR_DB_PATH', 'data_storage/embeddings')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')

        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', 'data_storage/bot.log')
        self.log_max_size_mb = int(os.getenv('LOG_MAX_SIZE_MB', '50'))
        self.log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))

        # Parse lists
        self._parse_lists()

        # Validate
        self._validate()

    def _parse_lists(self):
        """Parse comma-separated lists"""
        # Channel IDs
        if self.channel_ids_str:
            self.channel_ids = [
                int(cid.strip())
                for cid in self.channel_ids_str.split(',')
                if cid.strip()
            ]
        else:
            self.channel_ids = []

        # Admin user IDs
        if self.admin_user_ids_str:
            self.admin_user_ids = [
                int(uid.strip())
                for uid in self.admin_user_ids_str.split(',')
                if uid.strip()
            ]
        else:
            self.admin_user_ids = []

    def _validate(self):
        """Validate configuration"""
        errors = []

        if not self.bot_token:
            errors.append("DISCORD_BOT_TOKEN is required")

        if not self.server_id:
            errors.append("DISCORD_SERVER_ID is required")

        if not self.channel_ids:
            errors.append("DISCORD_CHANNEL_IDS is required (at least one channel)")

        if not self.admin_user_ids:
            errors.append("ADMIN_USER_IDS is required (at least one admin)")

        if self.response_rate < 0 or self.response_rate > 1:
            errors.append("RESPONSE_RATE must be between 0 and 1")

        if self.temperature < 0 or self.temperature > 2:
            errors.append("GENERATION_TEMPERATURE must be between 0 and 2")

        if errors:
            raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))

    def is_admin(self, user_id: int) -> bool:
        """
        Check if user is an admin

        Args:
            user_id: Discord user ID

        Returns:
            True if user is admin, False otherwise
        """
        return user_id in self.admin_user_ids

    def should_respond_to_channel(self, channel_id: int) -> bool:
        """
        Check if bot should monitor this channel

        Args:
            channel_id: Discord channel ID

        Returns:
            True if channel is monitored, False otherwise
        """
        return channel_id in self.channel_ids

    def __repr__(self) -> str:
        """String representation (hide sensitive data)"""
        return (
            f"BotConfig(\n"
            f"  bot_token={'*' * 8}{self.bot_token[-4:] if self.bot_token else 'None'},\n"
            f"  server_id={self.server_id},\n"
            f"  channels={len(self.channel_ids)},\n"
            f"  admins={len(self.admin_user_ids)},\n"
            f"  model_path={self.model_path},\n"
            f"  response_rate={self.response_rate * 100}%\n"
            f")"
        )


# Global config instance (load once)
_config: Optional[BotConfig] = None


def get_config(reload: bool = False) -> BotConfig:
    """
    Get global config instance

    Args:
        reload: Force reload from .env

    Returns:
        BotConfig instance
    """
    global _config

    if _config is None or reload:
        _config = BotConfig()

    return _config


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loading...")
    try:
        config = get_config()
        print("✅ Configuration loaded successfully!")
        print(config)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
