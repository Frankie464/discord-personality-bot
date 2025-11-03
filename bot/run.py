"""
24/7 Bot Runner - v2.0 Architecture

This is the main entry point for the Discord bot's 24/7 operation.

Key Features (v2.0):
- Singleton model loading (load once at startup, never reload)
- Watchdog heartbeat (for auto-restart on failure)
- Async inference (non-blocking Discord event loop)
- Channel allowlist awareness
- Minimal RAG context retrieval
- No message fetching (runs separately via fetch_and_embed.py)

Process Split:
- This bot runs 24/7 for RESPONSES ONLY
- Does NOT fetch message history (separate script)
- Focuses on low-latency, high-reliability operation
- Monitored by watchdog for auto-restart

Performance:
- Model load time: 15-20 seconds (once at startup)
- Response time: 2-3 seconds (target)
- Memory usage: 3-4GB (stable)
- Uptime target: 99%+

Usage:
    python bot/run.py

Or via GUI:
    python launcher.py  # Click "Start Bot"
"""

import os
import sys
import asyncio
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import discord
from discord.ext import commands

from model.inference import get_model, generate_response, is_model_loaded
from storage.database import init_database
from storage.vectordb import VectorDatabase
from bot.watchdog import update_heartbeat


class PersonalityBot(commands.Bot):
    """
    Discord Personality Bot - v2.0

    24/7 bot with singleton model loading and watchdog monitoring
    """

    def __init__(self, *args, **kwargs):
        """Initialize bot"""
        super().__init__(*args, **kwargs)

        # Configuration
        self.model_path = os.getenv('MODEL_PATH', 'models/finetuned/qwen2.5-3b-personality-q4.gguf')
        self.model_chat_template = os.getenv('MODEL_CHAT_TEMPLATE', 'chatml')
        self.model_context_length = int(os.getenv('MODEL_CONTEXT_LENGTH', '2048'))
        self.model_threads = int(os.getenv('MODEL_THREADS', '0'))
        self.gpu_layers = int(os.getenv('GPU_LAYERS', '0'))

        # Generation parameters (from database/config)
        self.db = None
        self.vector_db = None
        self.model = None
        self.model_loaded = False

        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'responses_sent': 0,
            'messages_seen': 0,
            'errors': 0,
            'last_response_time': None
        }

    async def setup_hook(self):
        """Called when bot is starting up (before ready)"""
        print("\n" + "=" * 70)
        print("DISCORD PERSONALITY BOT - v2.0 (24/7 Runner)")
        print("=" * 70)
        print(f"Starting: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Initialize database
        print("🗄️  Initializing database...")
        db_path = os.getenv('DATABASE_PATH', 'data_storage/database/bot.db')
        self.db = init_database(db_path)
        print("✅ Database ready")

        # Initialize vector database (optional for RAG)
        print("🧬 Initializing vector database (for RAG context)...")
        try:
            vector_db_path = os.getenv('VECTOR_DB_PATH', 'data_storage/embeddings')
            embedding_model = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
            self.vector_db = VectorDatabase(
                db_path=vector_db_path,
                embedding_model=embedding_model
            )
            print("✅ Vector database ready")
        except ImportError:
            print("⚠️  LanceDB not available, RAG context disabled")
            self.vector_db = None
        except Exception as e:
            print(f"⚠️  Vector database initialization failed: {e}")
            self.vector_db = None

        # Load model (SINGLETON - loads once, never reloads)
        print()
        print("🤖 Loading model (singleton pattern)...")
        print(f"   Model: {self.model_path}")
        print(f"   Chat template: {self.model_chat_template} (CRITICAL for Qwen2.5)")
        print(f"   Context length: {self.model_context_length}")
        print(f"   GPU layers: {self.gpu_layers} (0 = CPU only)")
        print(f"   Threads: {'auto' if self.model_threads == 0 else self.model_threads}")
        print()

        try:
            load_start = time.time()

            self.model = get_model(
                model_path=self.model_path,
                n_ctx=self.model_context_length,
                n_threads=self.model_threads,
                n_gpu_layers=self.gpu_layers,
                chat_format=self.model_chat_template,
                verbose=False
            )

            load_time = time.time() - load_start
            self.model_loaded = True

            print(f"✅ Model loaded in {load_time:.1f}s (will stay loaded)")
            print(f"   Memory: ~{os.path.getsize(self.model_path) / (1024**3):.1f}GB")

        except FileNotFoundError as e:
            print(f"❌ Model file not found: {e}")
            print("   Please train the model first or check MODEL_PATH in .env")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print()
        print("=" * 70)
        print("✅ BOT INITIALIZATION COMPLETE")
        print("=" * 70)
        print()

    async def on_ready(self):
        """Called when bot successfully connects to Discord"""
        print(f"🟢 Bot online: {self.user} (ID: {self.user.id})")
        print(f"   Connected to {len(self.guilds)} server(s)")
        print()

        # Start watchdog heartbeat task
        self.loop.create_task(self.heartbeat_task())

        print("🤖 Bot is ready to respond!")
        print(f"   Response rate: {self.db.get_config_value('response_rate', '0.05')}  ")
        print(f"   Temperature: {self.db.get_config_value('temperature', '0.7')}")
        print(f"   Max tokens: {self.db.get_config_value('max_tokens', '120')}")
        print()

    async def heartbeat_task(self):
        """Update heartbeat file periodically for watchdog"""
        while not self.is_closed():
            try:
                update_heartbeat()
            except Exception as e:
                print(f"⚠️  Heartbeat update failed: {e}")

            await asyncio.sleep(30)  # Update every 30 seconds

    async def on_message(self, message: discord.Message):
        """Handle incoming messages"""
        # Ignore bot's own messages
        if message.author == self.user:
            return

        # Ignore other bots
        if message.author.bot:
            return

        self.stats['messages_seen'] += 1

        # Check if should respond
        should_respond = await self._should_respond(message)

        if not should_respond:
            return

        # Generate response
        try:
            async with message.channel.typing():
                response = await self._generate_response_async(message)

            if response:
                await message.channel.send(response)
                self.stats['responses_sent'] += 1
                self.stats['last_response_time'] = datetime.now()

        except Exception as e:
            print(f"❌ Error responding to message: {e}")
            self.stats['errors'] += 1
            import traceback
            traceback.print_exc()

    async def _should_respond(self, message: discord.Message) -> bool:
        """Determine if bot should respond to this message"""
        # Always respond to mentions
        if self.user.mentioned_in(message):
            return True

        # Check RESPOND_ONLY_TO_MENTIONS setting
        respond_only_mentions = self.db.get_config_value('respond_only_to_mentions', 'false').lower() == 'true'
        if respond_only_mentions:
            return False  # Only respond to mentions

        # Random response based on response_rate
        import random
        response_rate = float(self.db.get_config_value('response_rate', '0.05'))
        return random.random() < response_rate

    async def _generate_response_async(self, message: discord.Message) -> Optional[str]:
        """
        Generate response using model (async execution)

        Uses asyncio.to_thread to run inference without blocking event loop
        """
        if not self.model_loaded:
            print("⚠️  Model not loaded, cannot generate response")
            return None

        # Get configuration
        temperature = float(self.db.get_config_value('temperature', '0.7'))
        top_p = float(self.db.get_config_value('top_p', '0.9'))
        top_k = int(self.db.get_config_value('top_k', '40'))
        max_tokens = int(self.db.get_config_value('max_tokens', '120'))
        repetition_penalty = float(self.db.get_config_value('repetition_penalty', '1.1'))

        # Get RAG context (optional)
        context = ""
        if self.vector_db is not None:
            try:
                context = self.vector_db.get_conversation_context(
                    current_message=message.content,
                    channel_id=str(message.channel.id),
                    limit=3
                )
            except Exception as e:
                print(f"⚠️  RAG context retrieval failed: {e}")

        # Build prompt
        system_prompt = "You're a regular on this Discord server. Chat naturally."

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add context if available
        if context:
            messages.append({
                "role": "system",
                "content": context
            })

        # Add user message
        messages.append({
            "role": "user",
            "content": message.content
        })

        # Generate response (run in thread pool to avoid blocking)
        start_time = time.time()

        response = await asyncio.to_thread(
            generate_response,
            self.model,
            messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )

        gen_time = time.time() - start_time
        print(f"💬 Response generated in {gen_time:.2f}s")

        return response

    def get_stats(self) -> dict:
        """Get bot statistics"""
        uptime = datetime.now() - self.stats['start_time']

        return {
            'uptime_seconds': uptime.total_seconds(),
            'messages_seen': self.stats['messages_seen'],
            'responses_sent': self.stats['responses_sent'],
            'errors': self.stats['errors'],
            'last_response_time': self.stats['last_response_time'].isoformat() if self.stats['last_response_time'] else None,
            'model_loaded': self.model_loaded
        }


def main():
    """Main entry point"""
    # Load environment
    load_dotenv()

    # Get bot token
    bot_token = os.getenv('DISCORD_BOT_TOKEN')
    if not bot_token:
        print("❌ Error: DISCORD_BOT_TOKEN not found in .env file")
        sys.exit(1)

    # Setup intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    intents.guilds = True

    # Create bot
    bot = PersonalityBot(
        command_prefix='!',  # For admin commands
        intents=intents
    )

    # Run bot
    try:
        bot.run(bot_token)
    except KeyboardInterrupt:
        print("\n\n⚠️  Bot stopped by user")
    except Exception as e:
        print(f"\n\n❌ Bot crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
