"""
Watchdog Module - 24/7 Bot Monitoring and Auto-Restart

This module provides reliability features for 24/7 operation:
- Health checks (ping every 30 seconds)
- Auto-restart on unresponsive bot
- Error recovery and graceful degradation
- Event logging for diagnostics

Architecture:
- Watchdog runs as separate process/thread
- Monitors bot process via heartbeat mechanism
- Restarts bot if heartbeat missing for >90 seconds
- Logs all restart events and reasons

Design Philosophy:
- Fail gracefully (log errors, don't crash)
- Auto-recover from transient failures
- Preserve state across restarts
- Alert on persistent issues
"""

from typing import Optional, Callable
import time
import threading
import subprocess
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
import signal


@dataclass
class HealthStatus:
    """Health check status data"""
    is_healthy: bool
    last_heartbeat: datetime
    consecutive_failures: int
    uptime_seconds: float
    restart_count: int
    last_error: Optional[str] = None


class BotWatchdog:
    """
    Watchdog monitor for Discord bot process

    Features:
    - Periodic health checks
    - Auto-restart on failure
    - Graceful shutdown handling
    - Event logging
    """

    def __init__(
        self,
        bot_script_path: str,
        check_interval: int = 30,
        failure_threshold: int = 3,
        restart_delay: int = 5,
        max_restarts_per_hour: int = 5,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize watchdog

        Args:
            bot_script_path: Path to bot main script (e.g., "bot/main.py")
            check_interval: Seconds between health checks (default: 30)
            failure_threshold: Failures before restart (default: 3)
            restart_delay: Seconds to wait before restart (default: 5)
            max_restarts_per_hour: Prevent restart loop (default: 5)
            log_callback: Optional function for logging (receives message string)
        """
        self.bot_script_path = bot_script_path
        self.check_interval = check_interval
        self.failure_threshold = failure_threshold
        self.restart_delay = restart_delay
        self.max_restarts_per_hour = max_restarts_per_hour
        self.log_callback = log_callback or print

        # State
        self.bot_process: Optional[subprocess.Popen] = None
        self.last_heartbeat = datetime.now()
        self.consecutive_failures = 0
        self.start_time = datetime.now()
        self.restart_count = 0
        self.restart_history: list[datetime] = []
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Heartbeat file (bot touches this periodically)
        self.heartbeat_file = "data_storage/bot_heartbeat.txt"
        os.makedirs(os.path.dirname(self.heartbeat_file), exist_ok=True)

    def _log(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [WATCHDOG] {message}"
        self.log_callback(formatted)

    def start_bot(self) -> bool:
        """
        Start the bot process

        Returns:
            True if started successfully, False otherwise
        """
        try:
            # Check if bot script exists
            if not os.path.exists(self.bot_script_path):
                self._log(f"❌ Bot script not found: {self.bot_script_path}")
                return False

            # Start bot as subprocess
            self._log(f"🚀 Starting bot: {self.bot_script_path}")

            self.bot_process = subprocess.Popen(
                [sys.executable, self.bot_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )

            # Reset state
            self.last_heartbeat = datetime.now()
            self.consecutive_failures = 0
            self.start_time = datetime.now()

            # Clear heartbeat file
            with open(self.heartbeat_file, 'w') as f:
                f.write(str(time.time()))

            self._log(f"✅ Bot started (PID: {self.bot_process.pid})")
            return True

        except Exception as e:
            self._log(f"❌ Failed to start bot: {e}")
            return False

    def stop_bot(self, timeout: int = 10) -> bool:
        """
        Stop the bot process gracefully

        Args:
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        if self.bot_process is None:
            self._log("⚠️  No bot process to stop")
            return True

        try:
            self._log("🛑 Stopping bot...")

            # Try graceful shutdown first (SIGTERM)
            self.bot_process.terminate()

            try:
                self.bot_process.wait(timeout=timeout)
                self._log("✅ Bot stopped gracefully")
                return True
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                self._log("⚠️  Bot not responding, forcing shutdown...")
                self.bot_process.kill()
                self.bot_process.wait(timeout=5)
                self._log("✅ Bot force-stopped")
                return True

        except Exception as e:
            self._log(f"❌ Failed to stop bot: {e}")
            return False

    def restart_bot(self) -> bool:
        """
        Restart the bot process

        Returns:
            True if restarted successfully, False otherwise
        """
        # Check restart rate limit
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_restarts = [
            dt for dt in self.restart_history
            if dt > one_hour_ago
        ]

        if len(recent_restarts) >= self.max_restarts_per_hour:
            self._log(
                f"❌ Restart rate limit exceeded "
                f"({len(recent_restarts)} restarts in last hour)"
            )
            return False

        # Stop bot
        self._log("🔄 Restarting bot...")
        self.stop_bot()

        # Wait before restart
        time.sleep(self.restart_delay)

        # Start bot
        success = self.start_bot()

        if success:
            self.restart_count += 1
            self.restart_history.append(datetime.now())
            self._log(f"✅ Bot restarted (total restarts: {self.restart_count})")
        else:
            self._log("❌ Bot restart failed")

        return success

    def check_health(self) -> HealthStatus:
        """
        Check bot health via heartbeat file

        Returns:
            HealthStatus object with current status
        """
        try:
            # Check if process still running
            if self.bot_process is None or self.bot_process.poll() is not None:
                return HealthStatus(
                    is_healthy=False,
                    last_heartbeat=self.last_heartbeat,
                    consecutive_failures=self.consecutive_failures + 1,
                    uptime_seconds=0,
                    restart_count=self.restart_count,
                    last_error="Process not running"
                )

            # Check heartbeat file
            if os.path.exists(self.heartbeat_file):
                mtime = os.path.getmtime(self.heartbeat_file)
                last_heartbeat_time = datetime.fromtimestamp(mtime)
                time_since_heartbeat = datetime.now() - last_heartbeat_time

                # Consider healthy if heartbeat within last 90 seconds
                is_healthy = time_since_heartbeat.total_seconds() < 90

                if is_healthy:
                    self.last_heartbeat = last_heartbeat_time
                    self.consecutive_failures = 0
                else:
                    self.consecutive_failures += 1

                uptime = (datetime.now() - self.start_time).total_seconds()

                return HealthStatus(
                    is_healthy=is_healthy,
                    last_heartbeat=last_heartbeat_time,
                    consecutive_failures=self.consecutive_failures,
                    uptime_seconds=uptime,
                    restart_count=self.restart_count,
                    last_error=None if is_healthy else "Heartbeat timeout"
                )

            # Heartbeat file missing
            self.consecutive_failures += 1
            return HealthStatus(
                is_healthy=False,
                last_heartbeat=self.last_heartbeat,
                consecutive_failures=self.consecutive_failures,
                uptime_seconds=0,
                restart_count=self.restart_count,
                last_error="Heartbeat file missing"
            )

        except Exception as e:
            self.consecutive_failures += 1
            return HealthStatus(
                is_healthy=False,
                last_heartbeat=self.last_heartbeat,
                consecutive_failures=self.consecutive_failures,
                uptime_seconds=0,
                restart_count=self.restart_count,
                last_error=str(e)
            )

    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        self._log("👀 Watchdog monitoring started")

        while self.is_running:
            try:
                # Check health
                health = self.check_health()

                if not health.is_healthy:
                    self._log(
                        f"⚠️  Health check failed "
                        f"({health.consecutive_failures}/{self.failure_threshold}): "
                        f"{health.last_error}"
                    )

                    # Restart if threshold exceeded
                    if health.consecutive_failures >= self.failure_threshold:
                        self._log(
                            f"❌ Bot unresponsive "
                            f"({health.consecutive_failures} consecutive failures)"
                        )
                        self.restart_bot()
                else:
                    # Log uptime periodically (every 10 checks = 5 minutes)
                    if int(health.uptime_seconds) % (self.check_interval * 10) < self.check_interval:
                        uptime_str = str(timedelta(seconds=int(health.uptime_seconds)))
                        self._log(f"✅ Bot healthy (uptime: {uptime_str})")

                # Wait before next check
                time.sleep(self.check_interval)

            except Exception as e:
                self._log(f"❌ Monitor loop error: {e}")
                time.sleep(self.check_interval)

        self._log("👋 Watchdog monitoring stopped")

    def start_monitoring(self) -> bool:
        """
        Start watchdog monitoring

        Returns:
            True if monitoring started, False otherwise
        """
        if self.is_running:
            self._log("⚠️  Watchdog already running")
            return False

        # Start bot if not running
        if self.bot_process is None or self.bot_process.poll() is not None:
            if not self.start_bot():
                return False

        # Start monitor thread
        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()

        self._log("✅ Watchdog monitoring enabled")
        return True

    def stop_monitoring(self):
        """Stop watchdog monitoring"""
        self._log("🛑 Stopping watchdog...")
        self.is_running = False

        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=10)

        self.stop_bot()
        self._log("✅ Watchdog stopped")

    def get_status(self) -> dict:
        """
        Get current watchdog status

        Returns:
            Dictionary with status information
        """
        health = self.check_health()

        return {
            'is_monitoring': self.is_running,
            'is_bot_healthy': health.is_healthy,
            'last_heartbeat': health.last_heartbeat.isoformat(),
            'consecutive_failures': health.consecutive_failures,
            'uptime_seconds': health.uptime_seconds,
            'uptime_formatted': str(timedelta(seconds=int(health.uptime_seconds))),
            'restart_count': health.restart_count,
            'restarts_last_hour': len([
                dt for dt in self.restart_history
                if dt > datetime.now() - timedelta(hours=1)
            ]),
            'last_error': health.last_error
        }


def update_heartbeat(heartbeat_file: str = "data_storage/bot_heartbeat.txt"):
    """
    Update heartbeat file (called by bot periodically)

    Args:
        heartbeat_file: Path to heartbeat file
    """
    try:
        os.makedirs(os.path.dirname(heartbeat_file), exist_ok=True)
        with open(heartbeat_file, 'w') as f:
            f.write(str(time.time()))
    except Exception as e:
        print(f"❌ Failed to update heartbeat: {e}")


if __name__ == "__main__":
    # Test watchdog (requires bot/main.py to exist)
    print("Testing Bot Watchdog...")
    print("=" * 60)

    # Create mock bot script for testing
    mock_bot_path = "test_mock_bot.py"
    with open(mock_bot_path, 'w') as f:
        f.write("""
import time
import os

# Mock bot that updates heartbeat
heartbeat_file = "data_storage/bot_heartbeat.txt"

print("Mock bot started")
os.makedirs(os.path.dirname(heartbeat_file), exist_ok=True)

while True:
    # Update heartbeat
    with open(heartbeat_file, 'w') as f:
        f.write(str(time.time()))
    print("Heartbeat updated")
    time.sleep(10)
""")

    print("\n1. Creating watchdog...")
    watchdog = BotWatchdog(
        bot_script_path=mock_bot_path,
        check_interval=5,  # Check every 5 seconds for testing
        failure_threshold=2,
        restart_delay=2
    )

    print("\n2. Starting monitoring...")
    watchdog.start_monitoring()

    print("\n3. Monitoring for 20 seconds...")
    time.sleep(20)

    print("\n4. Checking status...")
    status = watchdog.get_status()
    print("   Status:")
    for key, value in status.items():
        print(f"     {key}: {value}")

    print("\n5. Stopping monitoring...")
    watchdog.stop_monitoring()

    # Cleanup
    if os.path.exists(mock_bot_path):
        os.remove(mock_bot_path)
    if os.path.exists("data_storage/bot_heartbeat.txt"):
        os.remove("data_storage/bot_heartbeat.txt")

    print("\n" + "=" * 60)
    print("✅ Watchdog test completed!")
    print("\nUsage in Production:")
    print("  1. Watchdog monitors bot via heartbeat file")
    print("  2. Bot calls update_heartbeat() every 30 seconds")
    print("  3. Watchdog checks heartbeat every 30 seconds")
    print("  4. If 3 consecutive failures → auto-restart")
    print("  5. Rate limit: max 5 restarts per hour")
    print("\nIntegration:")
    print("  - GUI calls watchdog.start_monitoring() on 'Start Bot'")
    print("  - Bot calls update_heartbeat() in async loop")
    print("  - GUI displays watchdog.get_status() in dashboard")
