#!/usr/bin/env python3
"""
Phase 1 Testing Script

Tests all Phase 1 components:
- Database initialization
- Privacy manager
- Configuration loading
- Module imports

Run this to verify Phase 1 is working correctly before message collection.
"""

import sys
import os


def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 70)
    print("Testing Module Imports")
    print("=" * 70)

    tests = []

    # Core Python modules
    try:
        import json
        tests.append(("json", True, ""))
    except ImportError as e:
        tests.append(("json", False, str(e)))

    try:
        import sqlite3
        tests.append(("sqlite3", True, ""))
    except ImportError as e:
        tests.append(("sqlite3", False, str(e)))

    # External dependencies
    try:
        import discord
        tests.append((f"discord.py v{discord.__version__}", True, ""))
    except ImportError as e:
        tests.append(("discord.py", False, str(e)))

    try:
        import dotenv
        tests.append(("python-dotenv", True, ""))
    except ImportError as e:
        tests.append(("python-dotenv", False, str(e)))

    try:
        import tqdm
        tests.append(("tqdm", True, ""))
    except ImportError as e:
        tests.append(("tqdm", False, str(e)))

    # Project modules
    try:
        from storage.database import Database, init_database
        tests.append(("storage.database", True, ""))
    except ImportError as e:
        tests.append(("storage.database", False, str(e)))

    try:
        from data.privacy import PrivacyManager
        tests.append(("data.privacy", True, ""))
    except ImportError as e:
        tests.append(("data.privacy", False, str(e)))

    try:
        from data.fetcher import MessageFetcher
        tests.append(("data.fetcher", True, ""))
    except ImportError as e:
        tests.append(("data.fetcher", False, str(e)))

    try:
        from bot.config import BotConfig
        tests.append(("bot.config", True, ""))
    except ImportError as e:
        tests.append(("bot.config", False, str(e)))

    try:
        from bot.commands import AdminCommands
        tests.append(("bot.commands", True, ""))
    except ImportError as e:
        tests.append(("bot.commands", False, str(e)))

    # Print results
    for name, success, error in tests:
        if success:
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}: {error}")

    failed = sum(1 for _, success, _ in tests if not success)
    print(f"\nResult: {len(tests) - failed}/{len(tests)} passed")

    return failed == 0


def test_database():
    """Test database initialization and operations"""
    print("\n" + "=" * 70)
    print("Testing Database")
    print("=" * 70)

    try:
        from storage.database import init_database

        # Initialize test database
        db = init_database("data_storage/database/test_bot.db")
        print("  ✅ Database initialized")

        # Test configuration
        db.set_config("test_key", "test_value")
        value = db.get_config("test_key")
        assert value == "test_value", "Config set/get failed"
        print("  ✅ Configuration read/write")

        # Test exclusion
        db.add_excluded_user(
            user_id="123456789",
            username="test_user",
            reason="Testing",
            excluded_by_admin="admin"
        )
        assert db.is_user_excluded("123456789"), "Exclusion failed"
        print("  ✅ User exclusion")

        # Test statistics
        db.log_statistics(
            messages_seen=100,
            responses_sent=5,
            avg_response_time=2.5,
            errors=0
        )
        stats = db.get_statistics(hours=24)
        assert stats['messages_seen'] == 100, "Statistics failed"
        print("  ✅ Statistics logging")

        # Cleanup
        if os.path.exists("data_storage/database/test_bot.db"):
            os.remove("data_storage/database/test_bot.db")

        print("\n✅ All database tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_privacy():
    """Test privacy manager"""
    print("\n" + "=" * 70)
    print("Testing Privacy Manager")
    print("=" * 70)

    try:
        from storage.database import init_database
        from data.privacy import PrivacyManager, should_include_message

        # Initialize
        db = init_database("data_storage/database/test_privacy.db")
        privacy = PrivacyManager(db)
        print("  ✅ Privacy manager initialized")

        # Test exclusion
        privacy.exclude_user(
            user_id="999888777",
            username="test_excluded",
            reason="Testing",
            admin_id="admin"
        )
        assert privacy.is_user_excluded("999888777"), "Exclusion failed"
        print("  ✅ User exclusion")

        # Test filtering
        messages = [
            {'user_id': '999888777', 'content': 'Excluded'},
            {'user_id': '111222333', 'content': 'Included'},
        ]
        filtered = privacy.filter_messages(messages)
        assert len(filtered) == 1, "Message filtering failed"
        assert filtered[0]['user_id'] == '111222333', "Wrong message kept"
        print("  ✅ Message filtering")

        # Test quality filters
        test_cases = [
            ({'content': 'lol', 'bot': False, 'type': 0, 'user_id': '111'}, True),
            ({'content': 'test', 'bot': True, 'type': 0, 'user_id': '111'}, False),
        ]
        for message, expected in test_cases:
            result = should_include_message(message, privacy)
            assert result == expected, f"Quality filter failed for {message}"
        print("  ✅ Quality filtering")

        # Cleanup
        if os.path.exists("data_storage/database/test_privacy.db"):
            os.remove("data_storage/database/test_privacy.db")

        print("\n✅ All privacy tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Privacy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading (requires .env file)"""
    print("\n" + "=" * 70)
    print("Testing Configuration")
    print("=" * 70)

    # Check if .env exists
    if not os.path.exists(".env"):
        print("  ⚠️  .env file not found")
        print("     Copy .env.example to .env and configure it")
        print("     Skipping configuration test")
        return None

    try:
        from bot.config import BotConfig

        config = BotConfig()
        print("  ✅ Configuration loaded")

        # Validate required fields
        assert config.bot_token, "Bot token missing"
        assert config.server_id, "Server ID missing"
        assert config.channel_ids, "Channel IDs missing"
        assert config.admin_user_ids, "Admin user IDs missing"
        print("  ✅ Required fields present")

        # Validate ranges
        assert 0 <= config.response_rate <= 1, "Invalid response rate"
        assert 0 <= config.temperature <= 2, "Invalid temperature"
        print("  ✅ Parameter ranges valid")

        print(f"\n  Configuration summary:")
        print(f"    Server ID: {config.server_id}")
        print(f"    Channels: {len(config.channel_ids)}")
        print(f"    Admins: {len(config.admin_user_ids)}")
        print(f"    Response rate: {config.response_rate * 100}%")

        print("\n✅ Configuration test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that required directories exist"""
    print("\n" + "=" * 70)
    print("Testing Directory Structure")
    print("=" * 70)

    required_dirs = [
        "data_storage",
        "data_storage/messages",
        "data_storage/database",
        "data_storage/embeddings",
        "models",
        "models/base",
        "models/finetuned",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ (missing)")
            all_exist = False

    if all_exist:
        print("\n✅ All directories exist!")
    else:
        print("\n⚠️  Some directories missing (will be created automatically)")

    return all_exist


def main():
    """Run all tests"""
    print("\n")
    print("=" * 70)
    print("PHASE 1 TEST SUITE")
    print("Discord Personality Bot")
    print("=" * 70)
    print("\nThis script verifies that Phase 1 is set up correctly.")
    print("It tests:")
    print("  • Module imports (dependencies)")
    print("  • Database operations")
    print("  • Privacy manager")
    print("  • Configuration loading")
    print("  • Directory structure")
    print("")

    results = {}

    # Run tests
    results['imports'] = test_imports()
    results['database'] = test_database()
    results['privacy'] = test_privacy()
    results['directories'] = test_directory_structure()
    results['config'] = test_configuration()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        if result is True:
            print(f"  ✅ {name.upper()}: PASSED")
        elif result is False:
            print(f"  ❌ {name.upper()}: FAILED")
        elif result is None:
            print(f"  ⚠️  {name.upper()}: SKIPPED")

    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nPhase 1 setup is complete. You can now:")
        print("  1. Configure your .env file (if not done)")
        print("  2. Run: python scripts/1_fetch_all_history.py")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nPlease fix the failed tests before proceeding.")
        print("Refer to SETUP_GUIDE.md for detailed instructions.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
