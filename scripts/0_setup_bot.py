"""
Automated Discord Personality Bot Setup Script

This script automates the entire bot setup process. Just provide your Discord
bot credentials and it handles everything else:
- Creates .env configuration file
- Initializes SQLite database
- Sets up channel allowlist
- Validates all configurations
- Shows next steps

Usage:
    python scripts/0_setup_bot.py

Requirements:
    - Discord bot created (https://discord.com/developers/applications)
    - Bot invited to server (offline is fine)
    - Bot token, server ID, channel IDs, admin user ID ready
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(project_root)


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step: int, text: str):
    """Print a step indicator"""
    print(f"\n[Step {step}] {text}")
    print("-" * 70)


def print_success(text: str):
    """Print a success message"""
    print(f"✅ {text}")


def print_error(text: str):
    """Print an error message"""
    print(f"❌ ERROR: {text}")


def print_warning(text: str):
    """Print a warning message"""
    print(f"⚠️  WARNING: {text}")


def print_info(text: str):
    """Print an info message"""
    print(f"ℹ️  {text}")


def validate_discord_token(token: str) -> bool:
    """Validate Discord bot token format"""
    # Discord tokens have 3 parts separated by dots
    # Format: base64.timestamp.hmac
    parts = token.split('.')
    if len(parts) != 3:
        return False

    # First part should be base64-encoded bot ID
    if len(parts[0]) < 20:
        return False

    return True


def validate_snowflake(snowflake: str) -> bool:
    """Validate Discord snowflake ID (numeric, 17-19 digits)"""
    if not snowflake.isdigit():
        return False

    if len(snowflake) < 17 or len(snowflake) > 20:
        return False

    return True


def get_bot_token() -> str:
    """Prompt user for Discord bot token"""
    print("\n📋 Discord Bot Token")
    print("   Where to find: Discord Developer Portal > Your App > Bot > Token")
    print("   Format: Should look like: MTIzNDU2Nzg5.GhqWXY.AbCdEfGhIjKlMnOpQrStUvWxYz")
    print()

    while True:
        token = input("Enter your Discord bot token: ").strip()

        if not token:
            print_error("Token cannot be empty. Please try again.")
            continue

        if not validate_discord_token(token):
            print_error("Invalid token format. Should be 3 parts separated by dots.")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                sys.exit(1)
            continue

        # Double check
        print(f"\nToken starts with: {token[:20]}...")
        confirm = input("Is this correct? (y/n): ").strip().lower()
        if confirm == 'y':
            return token


def get_server_id() -> str:
    """Prompt user for Discord server ID"""
    print("\n🏠 Discord Server ID")
    print("   How to get: Right-click your server name > Copy Server ID")
    print("   (Enable Developer Mode in Discord Settings > Advanced if you don't see this)")
    print()

    while True:
        server_id = input("Enter your Discord server ID: ").strip()

        if not server_id:
            print_error("Server ID cannot be empty. Please try again.")
            continue

        if not validate_snowflake(server_id):
            print_error("Invalid server ID. Should be 17-19 digit number.")
            continue

        print(f"\nServer ID: {server_id}")
        confirm = input("Is this correct? (y/n): ").strip().lower()
        if confirm == 'y':
            return server_id


def get_channel_ids() -> List[str]:
    """Prompt user for Discord channel IDs"""
    print("\n📺 Discord Channel IDs")
    print("   How to get: Right-click each channel > Copy Channel ID")
    print("   Recommended: Active conversation channels (#general, #chat, etc.)")
    print("   Avoid: Bot command channels, admin channels, low-activity channels")
    print()
    print("   Enter channel IDs one at a time. Press Enter with empty input when done.")
    print("   Need at least 1 channel, recommended 3-5 channels.")
    print()

    channel_ids = []
    while True:
        if len(channel_ids) == 0:
            prompt = f"Enter channel ID #{len(channel_ids) + 1}: "
        else:
            prompt = f"Enter channel ID #{len(channel_ids) + 1} (or press Enter if done): "

        channel_id = input(prompt).strip()

        # If empty and we have at least one, we're done
        if not channel_id:
            if len(channel_ids) > 0:
                print(f"\n✅ {len(channel_ids)} channel(s) added: {', '.join(channel_ids)}")
                confirm = input("Continue with these channels? (y/n): ").strip().lower()
                if confirm == 'y':
                    return channel_ids
                else:
                    # Reset and start over
                    channel_ids = []
                    print("\nLet's start over. Enter channel IDs:")
                    continue
            else:
                print_error("Need at least 1 channel. Please enter a channel ID.")
                continue

        # Validate
        if not validate_snowflake(channel_id):
            print_error("Invalid channel ID. Should be 17-19 digit number.")
            continue

        # Check for duplicates
        if channel_id in channel_ids:
            print_warning("Channel ID already added. Skipping.")
            continue

        channel_ids.append(channel_id)
        print(f"   Added channel {channel_id}")


def get_admin_user_ids() -> List[str]:
    """Prompt user for admin user IDs"""
    print("\n👤 Admin User IDs")
    print("   How to get: Right-click your username in member list > Copy User ID")
    print("   Admins can use bot commands like !setrate, !settemp, !status, etc.")
    print()
    print("   Enter admin user IDs one at a time. Press Enter with empty input when done.")
    print("   Need at least 1 admin (yourself).")
    print()

    admin_ids = []
    while True:
        if len(admin_ids) == 0:
            prompt = f"Enter admin user ID #{len(admin_ids) + 1} (yourself): "
        else:
            prompt = f"Enter admin user ID #{len(admin_ids) + 1} (or press Enter if done): "

        admin_id = input(prompt).strip()

        # If empty and we have at least one, we're done
        if not admin_id:
            if len(admin_ids) > 0:
                print(f"\n✅ {len(admin_ids)} admin(s) added: {', '.join(admin_ids)}")
                confirm = input("Continue with these admins? (y/n): ").strip().lower()
                if confirm == 'y':
                    return admin_ids
                else:
                    # Reset and start over
                    admin_ids = []
                    print("\nLet's start over. Enter admin user IDs:")
                    continue
            else:
                print_error("Need at least 1 admin. Please enter your user ID.")
                continue

        # Validate
        if not validate_snowflake(admin_id):
            print_error("Invalid user ID. Should be 17-19 digit number.")
            continue

        # Check for duplicates
        if admin_id in admin_ids:
            print_warning("User ID already added. Skipping.")
            continue

        admin_ids.append(admin_id)
        print(f"   Added admin {admin_id}")


def create_env_file(token: str, server_id: str, channel_ids: List[str], admin_ids: List[str]):
    """Create .env configuration file"""
    print_step(2, "Creating .env configuration file")

    env_path = project_root / ".env"

    # Check if .env already exists
    if env_path.exists():
        print_warning(".env file already exists!")
        overwrite = input("Overwrite existing .env? (y/n): ").strip().lower()
        if overwrite != 'y':
            print_info("Keeping existing .env file. Skipping this step.")
            return

        # Backup existing .env
        backup_path = project_root / ".env.backup"
        env_path.rename(backup_path)
        print_info(f"Backed up existing .env to {backup_path}")

    # Create .env content
    env_content = f"""# Discord Bot Configuration
DISCORD_BOT_TOKEN={token}
DISCORD_SERVER_ID={server_id}
DISCORD_CHANNEL_IDS={','.join(channel_ids)}

# Admin User IDs (comma-separated)
ADMIN_USER_IDS={','.join(admin_ids)}

# Model Configuration (for later phases)
MODEL_PATH=models/finetuned/qwen2.5-3b-personality-q4.gguf
MODEL_CONTEXT_LENGTH=2048
MODEL_THREADS=0  # 0 = auto-detect

# Generation Parameters
GENERATION_TEMPERATURE=0.75
GENERATION_TOP_P=0.9
GENERATION_TOP_K=40
GENERATION_MAX_TOKENS=150
GENERATION_REPETITION_PENALTY=1.1

# Bot Behavior
RESPONSE_RATE=0.05  # 5% probability to respond
ALWAYS_RESPOND_TO_MENTIONS=true

# Vector Database
VECTOR_DB_PATH=data_storage/embeddings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Database
DATABASE_PATH=data_storage/database/bot.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=data_storage/bot.log
LOG_MAX_SIZE_MB=50
LOG_BACKUP_COUNT=5

# GUI Settings
GUI_START_BOT_ON_LAUNCH=false
GUI_START_ON_WINDOWS_BOOT=false
GUI_MINIMIZE_TO_TRAY=true
GUI_SHOW_NOTIFICATIONS=true
"""

    # Write .env file
    env_path.write_text(env_content, encoding='utf-8')
    print_success(f"Created .env file at {env_path}")


def initialize_database():
    """Initialize SQLite database"""
    print_step(3, "Initializing SQLite database")

    try:
        from storage.database import init_database

        db = init_database()
        print_success("Database initialized successfully!")

        # Check that database file exists
        db_path = project_root / "data_storage" / "database" / "bot.db"
        if db_path.exists():
            size_kb = db_path.stat().st_size / 1024
            print_info(f"Database created at {db_path} ({size_kb:.1f} KB)")

        return db

    except Exception as e:
        print_error(f"Failed to initialize database: {e}")
        print_info("You may need to install dependencies first:")
        print_info("  python -m pip install -r requirements.txt")
        sys.exit(1)


def setup_channel_allowlist(db, channel_ids: List[str]):
    """Set up channel allowlist in database"""
    print_step(4, "Setting up channel allowlist")

    try:
        cursor = db.cursor()

        # Add channels to allowlist
        for channel_id in channel_ids:
            cursor.execute("""
                INSERT OR IGNORE INTO channel_allowlist (channel_id, added_at)
                VALUES (?, datetime('now'))
            """, (channel_id,))

        db.commit()

        # Verify
        cursor.execute("SELECT COUNT(*) FROM channel_allowlist")
        count = cursor.fetchone()[0]

        print_success(f"Added {len(channel_ids)} channel(s) to allowlist")
        print_info(f"Total channels in allowlist: {count}")

    except Exception as e:
        print_error(f"Failed to set up channel allowlist: {e}")
        sys.exit(1)


def verify_configuration():
    """Verify that configuration is valid"""
    print_step(5, "Verifying configuration")

    try:
        from bot.config import load_config

        config = load_config()
        print_success("Configuration loaded successfully!")

        # Print summary
        print("\n📋 Configuration Summary:")
        print(f"   Server ID: {config.server_id}")
        print(f"   Channels: {len(config.channel_ids)}")
        print(f"   Admins: {len(config.admin_user_ids)}")
        print(f"   Response Rate: {config.response_rate * 100:.1f}%")
        print(f"   Model Path: {config.model_path}")

        return True

    except Exception as e:
        print_error(f"Configuration validation failed: {e}")
        return False


def show_next_steps():
    """Show next steps to user"""
    print_header("🎉 Setup Complete!")

    print("Your Discord bot is now configured and ready for the next steps.\n")

    print("Next Steps:")
    print()
    print("1️⃣  COLLECT MESSAGE HISTORY (Bot stays OFFLINE)")
    print("   Run: python scripts/fetch_and_embed.py")
    print("   This fetches your server's message history for training.")
    print("   Time: 10-60 minutes depending on server size")
    print("   Target: 20,000+ messages")
    print()
    print("2️⃣  PREPARE TRAINING DATA")
    print("   Run: python scripts/2_prepare_training_data.py")
    print("   This formats messages for model training.")
    print("   Time: 2-5 minutes")
    print()
    print("3️⃣  TRAIN MODEL (Requires RTX 3070 8GB or better)")
    print("   Run: python scripts/3_train_model.py --mode sft+dpo")
    print("   This fine-tunes the model on your server's personality.")
    print("   Time: 5-7 hours on RTX 3070")
    print()
    print("4️⃣  EVALUATE PERSONALITY")
    print("   Run: python scripts/4_evaluate_personality.py")
    print("   This tests how well the bot matches your server's style.")
    print("   Time: 5-10 minutes")
    print()
    print("5️⃣  DEPLOY BOT (Bot comes ONLINE)")
    print("   Run: python bot/run.py")
    print("   Your bot will come online and start responding!")
    print()
    print("=" * 70)
    print("\n💡 TIP: Steps 1-4 can be done with bot OFFLINE. Only step 5 brings")
    print("   the bot online in your Discord server.")
    print()
    print("📖 For detailed instructions, see:")
    print("   - SETUP_GUIDE.md - Complete setup walkthrough")
    print("   - CLAUDE.md - Technical documentation")
    print("   - TODO.md - Full implementation checklist")
    print()


def main():
    """Main setup flow"""
    print_header("Discord Personality Bot - Automated Setup")

    print("This script will guide you through setting up your Discord personality bot.")
    print("You'll need to have:")
    print("  ✅ Created a Discord bot (https://discord.com/developers/applications)")
    print("  ✅ Invited bot to your server (can be offline)")
    print("  ✅ Enabled Message Content Intent in bot settings")
    print("  ✅ Your bot token, server ID, channel IDs, and admin user ID ready")
    print()

    ready = input("Ready to begin? (y/n): ").strip().lower()
    if ready != 'y':
        print("\nSetup cancelled. Run this script again when you're ready!")
        sys.exit(0)

    # Step 1: Collect information
    print_step(1, "Collecting Discord configuration")

    token = get_bot_token()
    server_id = get_server_id()
    channel_ids = get_channel_ids()
    admin_ids = get_admin_user_ids()

    print_success("All information collected!")

    # Step 2: Create .env file
    create_env_file(token, server_id, channel_ids, admin_ids)

    # Step 3: Initialize database
    db = initialize_database()

    # Step 4: Setup channel allowlist
    setup_channel_allowlist(db, channel_ids)

    # Step 5: Verify configuration
    if not verify_configuration():
        print_error("Setup completed but configuration validation failed.")
        print_info("Check error messages above and try running setup again.")
        sys.exit(1)

    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user. Run script again to continue.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
