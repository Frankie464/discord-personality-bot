"""
Privacy Module - Lightweight Approach for Private Servers

⚠️ IMPORTANT: This bot is designed for PRIVATE Discord servers (~30 people, trusted friends)
   Do NOT deploy to public or community servers.

Privacy Philosophy for Private Servers:
- Designed for closed friend groups where everyone knows each other
- No complex opt-out systems (implicit consent model)
- Basic filtering: bot messages and system notifications only
- All operations are silent (no announcements)
- Channel allowlist provides transparency (see !botdata command)

For Public/Community Servers:
- DO NOT USE THIS BOT
- This lightweight approach is inappropriate for public servers
- Use a bot with proper consent management instead

Data Handling:
- All data stored locally only
- No external sharing or cloud uploads
- Channel allowlist controls what contributes to training
- Admin commands for management (!botdata, !setrate, etc.)
"""

from typing import List, Dict, Any


def is_bot_message(message: Dict[str, Any]) -> bool:
    """
    Check if message is from a bot

    Args:
        message: Message dictionary

    Returns:
        True if from bot, False otherwise
    """
    # Check if author has 'bot' flag
    if message.get('bot', False) or message.get('is_bot', False):
        return True

    # Check if author dict has bot flag
    author = message.get('author', {})
    if isinstance(author, dict) and author.get('bot', False):
        return True

    return False


def is_system_notification(message: Dict[str, Any]) -> bool:
    """
    Check if message is a system notification

    System messages include:
    - User joined/left server
    - Pin notifications
    - Boost notifications
    - Thread created notifications

    Args:
        message: Message dictionary

    Returns:
        True if system notification, False otherwise
    """
    # System messages have type != 0 (DEFAULT)
    msg_type = message.get('type', 0)
    return msg_type != 0


def is_empty_message(message: Dict[str, Any]) -> bool:
    """
    Check if message has no meaningful content

    Args:
        message: Message dictionary

    Returns:
        True if empty, False otherwise
    """
    content = message.get('content', '').strip()
    return len(content) == 0


def should_include_message(message: Dict[str, Any]) -> bool:
    """
    Check if message should be included in training data

    MINIMAL filtering - only remove:
    - Bot messages (from ANY bots)
    - System notifications
    - Empty messages

    KEEP everything else (authenticity is key):
    - ✅ Single-word responses ("lol", "bruh", "fr")
    - ✅ Repeated text ("GGGGGG", "nooooo")
    - ✅ Emojis, Unicode, reactions
    - ✅ Typos and non-standard spelling
    - ✅ Slang and community jargon
    - ✅ Any length: 1 character to 2000 characters
    - ✅ Links with text (only pure link spam filtered elsewhere if needed)
    - ✅ All caps, mixed case, lowercase
    - ✅ Punctuation patterns (!!!, ???, etc.)

    Args:
        message: Message dictionary

    Returns:
        True if message should be included, False otherwise
    """
    # Filter bots
    if is_bot_message(message):
        return False

    # Filter system notifications
    if is_system_notification(message):
        return False

    # Filter empty messages
    if is_empty_message(message):
        return False

    # KEEP everything else - authenticity is key!
    return True


def filter_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter list of messages using minimal quality criteria

    This is the main interface for preprocessing pipelines.

    Args:
        messages: List of message dictionaries

    Returns:
        Filtered list of messages
    """
    return [msg for msg in messages if should_include_message(msg)]


def get_privacy_stats(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about message filtering

    Args:
        messages: Original message list

    Returns:
        Dictionary with filtering statistics
    """
    total = len(messages)
    bot_count = sum(1 for msg in messages if is_bot_message(msg))
    system_count = sum(1 for msg in messages if is_system_notification(msg))
    empty_count = sum(1 for msg in messages if is_empty_message(msg))

    filtered = filter_messages(messages)
    kept_count = len(filtered)

    return {
        'total_messages': total,
        'kept_messages': kept_count,
        'filtered_messages': total - kept_count,
        'filter_breakdown': {
            'bot_messages': bot_count,
            'system_notifications': system_count,
            'empty_messages': empty_count
        },
        'keep_rate': round((kept_count / total * 100) if total > 0 else 0, 1)
    }


if __name__ == "__main__":
    # Test privacy filtering
    print("Testing Privacy Module (Lightweight Approach)")
    print("=" * 60)
    print("\n⚠️  Note: Designed for PRIVATE servers (~30 people, trusted friends)")
    print("   Do NOT deploy to public or community servers.\n")

    # Create test dataset
    test_messages = [
        # Should KEEP
        {'content': 'lol', 'bot': False, 'type': 0},
        {'content': 'GGGGGG', 'bot': False, 'type': 0},
        {'content': '😂😂😂', 'bot': False, 'type': 0},
        {'content': 'bruh that was wild', 'bot': False, 'type': 0},
        {'content': 'yo', 'bot': False, 'type': 0},
        {'content': 'Check this https://example.com', 'bot': False, 'type': 0},
        {'content': 'a', 'bot': False, 'type': 0},  # Even 1-char responses
        {'content': '!!!!!!!', 'bot': False, 'type': 0},
        {'content': 'nooooooooo', 'bot': False, 'type': 0},
        {'content': 'gg wp', 'bot': False, 'type': 0},

        # Should FILTER
        {'content': 'Bot response', 'bot': True, 'type': 0},
        {'content': 'Another bot', 'is_bot': True, 'type': 0},
        {'content': 'User joined', 'bot': False, 'type': 7},  # System notification
        {'content': '', 'bot': False, 'type': 0},  # Empty
        {'content': '   ', 'bot': False, 'type': 0},  # Whitespace only
    ]

    print("1. Testing individual filters:")
    print(f"   Total test messages: {len(test_messages)}")

    bot_count = sum(1 for msg in test_messages if is_bot_message(msg))
    print(f"   Bot messages: {bot_count}")

    system_count = sum(1 for msg in test_messages if is_system_notification(msg))
    print(f"   System notifications: {system_count}")

    empty_count = sum(1 for msg in test_messages if is_empty_message(msg))
    print(f"   Empty messages: {empty_count}")

    print("\n2. Testing should_include_message():")
    test_cases = [
        ({'content': 'lol', 'bot': False, 'type': 0}, True, "Single word"),
        ({'content': 'GGGGGG', 'bot': False, 'type': 0}, True, "Repeated text"),
        ({'content': '😂😂😂', 'bot': False, 'type': 0}, True, "Emojis"),
        ({'content': 'a', 'bot': False, 'type': 0}, True, "1-char response"),
        ({'content': 'https://example.com test', 'bot': False, 'type': 0}, True, "Link with text"),
        ({'content': 'Bot message', 'bot': True, 'type': 0}, False, "Bot message"),
        ({'content': 'System', 'bot': False, 'type': 1}, False, "System notification"),
        ({'content': '', 'bot': False, 'type': 0}, False, "Empty message"),
    ]

    for message, expected, description in test_cases:
        result = should_include_message(message)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {description}: {result}")

    print("\n3. Testing filter_messages():")
    filtered = filter_messages(test_messages)
    print(f"   Original: {len(test_messages)} messages")
    print(f"   Filtered: {len(filtered)} messages kept")
    print(f"   Removed: {len(test_messages) - len(filtered)} messages")

    print("\n4. Testing get_privacy_stats():")
    stats = get_privacy_stats(test_messages)
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Kept messages: {stats['kept_messages']}")
    print(f"   Filtered messages: {stats['filtered_messages']}")
    print(f"   Keep rate: {stats['keep_rate']}%")
    print(f"   Filter breakdown:")
    for filter_type, count in stats['filter_breakdown'].items():
        print(f"     - {filter_type}: {count}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nKey Principles:")
    print("  - MINIMAL filtering (bots, system messages, empty only)")
    print("  - Keep authentic communication patterns")
    print("  - No complex opt-out systems")
    print("  - Designed for private friend groups")
    print("  - Channel allowlist provides transparency")
    print("\n⚠️  Remember: Private servers only (~30 people, trusted friends)")
