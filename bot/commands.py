"""
Admin Commands for Discord Personality Bot - v2.0

All commands are admin-only.

Key Commands (v2.0):
- !botdata: Show channel allowlist (transparency)
- !setrate: Adjust response rate
- !settemp: Adjust temperature
- !setmaxlen: Adjust max response length
- !status: Show bot status and statistics
- !fetch: Manually trigger incremental message fetch
- !restart: Restart bot process
- !help: Show admin commands

Removed from v1.0:
- !exclude/!unexclude/!excluded (simplified privacy for private servers)
"""

import discord
from discord.ext import commands
import asyncio
import subprocess
import sys
import os
from typing import Optional

from bot.config import BotConfig
from storage.database import Database


class AdminCommands(commands.Cog):
    """Admin-only command handlers for v2.0"""

    def __init__(
        self,
        bot: commands.Bot,
        config: BotConfig,
        database: Database
    ):
        """
        Initialize admin commands

        Args:
            bot: Discord bot instance
            config: Bot configuration
            database: Database instance
        """
        self.bot = bot
        self.config = config
        self.db = database

    def is_admin(self, user_id: int) -> bool:
        """Check if user is admin"""
        return self.config.is_admin(user_id)

    @commands.command(name='botdata')
    async def show_training_data_sources(self, ctx: commands.Context):
        """
        Show which channels contribute to bot's training data (transparency)

        This command provides transparency about what data shapes the bot's
        personality. Shows the channel allowlist.

        Usage:
            !botdata
        """
        # Admin check
        if not self.is_admin(ctx.author.id):
            await ctx.send("❌ Admin only command")
            return

        # Get allowlisted channels
        allowed_channels = self.db.get_allowed_channels(enabled_only=False)

        if not allowed_channels:
            await ctx.send(
                "⚠️  **No channels in allowlist**\n\n"
                "No channels are currently contributing to training data.\n"
                "Add channels using database methods or GUI."
            )
            return

        # Format message
        enabled_channels = [ch for ch in allowed_channels if ch.get('enabled', True)]
        disabled_channels = [ch for ch in allowed_channels if not ch.get('enabled', True)]

        msg = f"**📊 Bot Training Data Sources**\n\n"
        msg += f"**Enabled Channels** ({len(enabled_channels)}):\n"

        if enabled_channels:
            for ch in enabled_channels:
                last_fetch = ch.get('last_fetch_at', 'Never')
                msg += f"• {ch['channel_name']} (ID: {ch['channel_id']})\n"
                msg += f"  Last fetch: {last_fetch}\n"
        else:
            msg += "  (none)\n"

        if disabled_channels:
            msg += f"\n**Disabled Channels** ({len(disabled_channels)}):\n"
            for ch in disabled_channels:
                msg += f"• {ch['channel_name']} (ID: {ch['channel_id']})\n"

        msg += "\n**Note:** Only enabled channels contribute to training data."
        msg += "\nMessages from these channels are used to train the bot's personality."

        await ctx.send(msg)

    @commands.command(name='setrate')
    async def set_response_rate(self, ctx: commands.Context, rate: float):
        """
        Admin-only: Set response rate

        Usage:
            !setrate 0.05  (5% response rate)
            !setrate 0.1   (10% response rate)

        Range: 0.0 - 1.0 (0% - 100%)
        """
        # Admin check
        if not self.is_admin(ctx.author.id):
            await ctx.send("❌ Admin only command")
            return

        # Validate
        if rate < 0 or rate > 1:
            await ctx.send("❌ Rate must be between 0.0 and 1.0")
            return

        # Update config
        self.db.set_config('response_rate', str(rate))
        self.config.response_rate = rate

        await ctx.send(f"✅ Response rate set to {rate * 100:.1f}%")

    @commands.command(name='settemp')
    async def set_temperature(self, ctx: commands.Context, temp: float):
        """
        Admin-only: Set generation temperature

        Usage:
            !settemp 0.7   (balanced - default)
            !settemp 0.5   (more focused)
            !settemp 0.9   (more creative)

        Range: 0.5 - 1.0 (recommended)
        """
        # Admin check
        if not self.is_admin(ctx.author.id):
            await ctx.send("❌ Admin only command")
            return

        # Validate
        if temp < 0.5 or temp > 1.0:
            await ctx.send("❌ Temperature must be between 0.5 and 1.0")
            return

        # Update config
        self.db.set_config('temperature', str(temp))
        self.config.temperature = temp

        await ctx.send(f"✅ Temperature set to {temp}")

    @commands.command(name='setmaxlen')
    async def set_max_length(self, ctx: commands.Context, length: int):
        """
        Admin-only: Set maximum response length

        Usage:
            !setmaxlen 120  (typical - default)
            !setmaxlen 80   (shorter)
            !setmaxlen 200  (longer)

        Range: 50 - 300 tokens
        """
        # Admin check
        if not self.is_admin(ctx.author.id):
            await ctx.send("❌ Admin only command")
            return

        # Validate
        if length < 50 or length > 300:
            await ctx.send("❌ Length must be between 50 and 300")
            return

        # Update config
        self.db.set_config('max_tokens', str(length))
        self.config.max_tokens = length

        await ctx.send(f"✅ Max response length set to {length} tokens")

    @commands.command(name='status')
    async def show_status(self, ctx: commands.Context):
        """
        Admin-only: Show bot status and statistics

        Usage:
            !status
        """
        # Admin check
        if not self.is_admin(ctx.author.id):
            await ctx.send("❌ Admin only command")
            return

        # Get statistics
        stats_24h = self.db.get_statistics(hours=24)
        stats_7d = self.db.get_statistics(hours=168)  # 7 days
        config = self.db.get_all_config()

        # Get allowlist info
        allowed_channels = self.db.get_allowed_channels(enabled_only=True)

        # Get bot stats if available
        bot_stats = {}
        if hasattr(self.bot, 'get_stats'):
            bot_stats = self.bot.get_stats()

        # Format status
        status_msg = (
            f"**🤖 Bot Status (v2.0)**\n\n"
            f"**Configuration:**\n"
            f"• Response Rate: {float(config.get('response_rate', 0.05)) * 100:.1f}%\n"
            f"• Temperature: {config.get('temperature', '0.7')}\n"
            f"• Max Tokens: {config.get('max_tokens', '120')}\n"
            f"• GPU Layers: {config.get('gpu_layers', '0')}\n"
            f"• Chat Template: {config.get('model_chat_template', 'chatml')}\n\n"
            f"**Channel Allowlist:**\n"
            f"• Enabled channels: {len(allowed_channels)}\n"
            f"• Use !botdata to see details\n\n"
        )

        if bot_stats:
            uptime_hours = bot_stats.get('uptime_seconds', 0) / 3600
            status_msg += (
                f"**Current Session:**\n"
                f"• Uptime: {uptime_hours:.1f} hours\n"
                f"• Messages seen: {bot_stats.get('messages_seen', 0):,}\n"
                f"• Responses sent: {bot_stats.get('responses_sent', 0):,}\n"
                f"• Model loaded: {bot_stats.get('model_loaded', False)}\n\n"
            )

        status_msg += (
            f"**Last 24 Hours:**\n"
            f"• Messages Seen: {stats_24h.get('messages_seen', 0):,}\n"
            f"• Responses Sent: {stats_24h.get('responses_sent', 0):,}\n"
            f"• Actual Rate: {stats_24h.get('response_rate', 0):.1f}%\n"
            f"• Avg Response Time: {stats_24h.get('avg_response_time', 0):.2f}s\n"
            f"• Errors: {stats_24h.get('errors', 0)}\n\n"
            f"**Last 7 Days:**\n"
            f"• Messages Seen: {stats_7d.get('messages_seen', 0):,}\n"
            f"• Responses Sent: {stats_7d.get('responses_sent', 0):,}\n"
            f"• Actual Rate: {stats_7d.get('response_rate', 0):.1f}%\n"
        )

        await ctx.send(status_msg)

    @commands.command(name='fetch')
    async def manual_fetch(self, ctx: commands.Context):
        """
        Admin-only: Manually trigger incremental message fetch

        This runs scripts/fetch_and_embed.py to fetch new messages
        and generate embeddings.

        Usage:
            !fetch
        """
        # Admin check
        if not self.is_admin(ctx.author.id):
            await ctx.send("❌ Admin only command")
            return

        await ctx.send("🔄 Starting incremental fetch and embed...")

        try:
            # Run fetch script
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'scripts',
                'fetch_and_embed.py'
            )

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                await ctx.send("✅ Fetch completed successfully!")
            else:
                await ctx.send(
                    f"⚠️  Fetch completed with errors:\n"
                    f"```\n{result.stderr[:1000]}\n```"
                )

        except subprocess.TimeoutExpired:
            await ctx.send("❌ Fetch timed out (>5 minutes)")
        except Exception as e:
            await ctx.send(f"❌ Fetch failed: {e}")

    @commands.command(name='restart')
    async def restart_bot(self, ctx: commands.Context):
        """
        Admin-only: Restart bot process

        Usage:
            !restart
        """
        # Admin check
        if not self.is_admin(ctx.author.id):
            await ctx.send("❌ Admin only command")
            return

        await ctx.send("🔄 Restarting bot... (watchdog will restart if enabled)")

        # Close bot gracefully
        await self.bot.close()

        # Exit process (watchdog or GUI will restart)
        sys.exit(0)

    @commands.command(name='help')
    async def show_help(self, ctx: commands.Context):
        """
        Admin-only: Show available commands

        Usage:
            !help
        """
        # Admin check
        if not self.is_admin(ctx.author.id):
            return

        help_msg = (
            f"**🤖 Admin Commands (v2.0)**\n\n"
            f"**Configuration:**\n"
            f"• `!setrate <0.0-1.0>` - Set response rate (default: 0.05)\n"
            f"• `!settemp <0.5-1.0>` - Set temperature (default: 0.7)\n"
            f"• `!setmaxlen <50-300>` - Set max response length (default: 120)\n\n"
            f"**Data Management:**\n"
            f"• `!botdata` - Show channel allowlist (transparency)\n"
            f"• `!fetch` - Manually trigger incremental fetch\n\n"
            f"**Information:**\n"
            f"• `!status` - Show bot status and statistics\n"
            f"• `!restart` - Restart bot process\n"
            f"• `!help` - Show this help\n\n"
            f"**v2.0 Changes:**\n"
            f"• Removed privacy commands (simplified for private servers)\n"
            f"• Added !botdata for channel allowlist transparency\n"
            f"• Added !fetch for manual data collection\n"
        )

        await ctx.send(help_msg)


async def setup(bot: commands.Bot):
    """Setup function for loading cog"""
    pass


if __name__ == "__main__":
    print("Admin Commands Module - v2.0")
    print("=" * 60)
    print("\nAdmin-only commands for Discord Personality Bot:")
    print("  • !botdata - Show channel allowlist (NEW)")
    print("  • !setrate - Adjust response rate")
    print("  • !settemp - Adjust temperature")
    print("  • !setmaxlen - Adjust max response length")
    print("  • !status - Show bot status")
    print("  • !fetch - Manual incremental fetch (NEW)")
    print("  • !restart - Restart bot (NEW)")
    print("  • !help - Show commands")
    print("\nAll commands require admin privileges.")
    print("\nv2.0 Changes:")
    print("  - Removed !exclude/!unexclude/!excluded commands")
    print("  - Simplified for private server use (~30 people)")
    print("  - Channel allowlist provides transparency")
    print("=" * 60)
