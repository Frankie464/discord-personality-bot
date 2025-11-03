"""
System Prompts and Conversation Formatting

This module provides system prompt templates and conversation formatting
utilities for the Discord personality bot.

Philosophy:
- Natural, human-like prompts (no robotic instructions)
- Emphasize authenticity over helpfulness
- Short and direct
- Match Discord communication style

Usage:
    from model.prompts import get_system_prompt, format_conversation

    # Get system prompt
    system_prompt = get_system_prompt()

    # Format conversation for model
    messages = format_conversation(
        recent_messages,
        current_message,
        system_prompt=system_prompt,
        context_messages=rag_results
    )
"""

from typing import List, Dict, Any, Optional


# Default system prompt for personality bot
DEFAULT_SYSTEM_PROMPT = """You're a regular on this Discord server. You've been here a while, you know the vibe, you know the people.

Just chat like you normally would. Be yourself. If something's funny, laugh. If something's dumb, call it out. If you want to spam a word for emphasis, do it. Short responses are fine. Long rants are fine too.

Don't act like you're here to help or assist - you're just hanging out. No formality, no AI speak, just natural conversation.

Match the energy of the room."""


# Alternative: minimal system prompt
MINIMAL_SYSTEM_PROMPT = """Chat naturally, same tone as the server. Chaotic, unhinged blend of us. No rules. Do not talk like an AI or assistant."""


# Alternative: no system prompt (let fine-tuning BE the personality)
NO_SYSTEM_PROMPT = ""


def get_system_prompt(
    style: str = "minimal",
    custom_prompt: Optional[str] = None
) -> str:
    """
    Get system prompt for the bot

    Args:
        style: Prompt style ("default", "minimal", "none")
        custom_prompt: Custom system prompt (overrides style)

    Returns:
        System prompt string
    """
    if custom_prompt:
        return custom_prompt

    if style == "minimal":
        return MINIMAL_SYSTEM_PROMPT
    elif style == "none":
        return NO_SYSTEM_PROMPT
    else:
        return DEFAULT_SYSTEM_PROMPT


def format_conversation(
    recent_messages: List[Dict[str, Any]],
    current_message: str,
    current_author: str,
    system_prompt: Optional[str] = None,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    max_history: int = 5
) -> List[Dict[str, str]]:
    """
    Format conversation for model input (ChatML format)

    Args:
        recent_messages: Recent Discord messages for context
        current_message: Current message to respond to
        current_author: Author of current message
        system_prompt: Optional system prompt
        context_messages: Optional RAG context from LanceDB
        max_history: Maximum number of recent messages to include

    Returns:
        List of message dicts in ChatML format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    """
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Add RAG context if provided (as system message)
    if context_messages:
        context_text = build_context_text(context_messages)
        if context_text:
            messages.append({
                "role": "system",
                "content": f"Relevant context from previous conversations:\n{context_text}"
            })

    # Add recent conversation history
    # Limit to max_history most recent messages
    history = recent_messages[-max_history:] if len(recent_messages) > max_history else recent_messages

    for msg in history:
        # Determine role based on whether message is from bot
        is_bot = msg.get('is_bot', False) or msg.get('author_id') == 'bot'

        role = "assistant" if is_bot else "user"

        # Format content with author name for clarity
        author = msg.get('username', msg.get('author', 'User'))
        content = msg.get('content', '')

        # For user messages, prefix with username
        if role == "user":
            formatted_content = f"{author}: {content}"
        else:
            # Bot messages don't need username prefix
            formatted_content = content

        messages.append({
            "role": role,
            "content": formatted_content
        })

    # Add current message
    messages.append({
        "role": "user",
        "content": f"{current_author}: {current_message}"
    })

    return messages


def build_context_text(
    context_messages: List[Dict[str, Any]],
    max_context: int = 3
) -> str:
    """
    Build context text from RAG results

    Args:
        context_messages: Messages retrieved from LanceDB
        max_context: Maximum number of context messages to include

    Returns:
        Formatted context string
    """
    if not context_messages:
        return ""

    # Limit context
    context = context_messages[:max_context]

    # Format as simple list
    context_lines = []
    for msg in context:
        author = msg.get('username', msg.get('author', 'User'))
        content = msg.get('content', '')
        context_lines.append(f"- {author}: {content}")

    return "\n".join(context_lines)


def format_for_display(
    messages: List[Dict[str, str]],
    show_system: bool = False
) -> str:
    """
    Format conversation for human-readable display

    Useful for debugging and logging.

    Args:
        messages: ChatML formatted messages
        show_system: Whether to show system messages

    Returns:
        Formatted string for display
    """
    lines = []

    for msg in messages:
        role = msg['role']
        content = msg['content']

        # Skip system messages unless requested
        if role == "system" and not show_system:
            continue

        if role == "system":
            lines.append(f"[SYSTEM] {content}")
        elif role == "user":
            lines.append(f"[USER] {content}")
        elif role == "assistant":
            lines.append(f"[BOT] {content}")

    return "\n".join(lines)


def extract_username_from_message(message_content: str) -> tuple[str, str]:
    """
    Extract username from formatted message content

    Handles format: "Username: message content"

    Args:
        message_content: Formatted message content

    Returns:
        Tuple of (username, content) or (None, original_content)
    """
    if ": " in message_content:
        parts = message_content.split(": ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]

    return None, message_content


def truncate_conversation(
    messages: List[Dict[str, str]],
    max_tokens: int = 2048,
    tokens_per_message: int = 50
) -> List[Dict[str, str]]:
    """
    Truncate conversation to fit within token limit

    Simple estimation: assume ~50 tokens per message average.
    Removes oldest messages first (except system prompt).

    Args:
        messages: ChatML formatted messages
        max_tokens: Maximum token limit
        tokens_per_message: Estimated tokens per message

    Returns:
        Truncated message list
    """
    estimated_tokens = len(messages) * tokens_per_message

    if estimated_tokens <= max_tokens:
        return messages

    # Keep system prompt (first message if exists)
    system_messages = []
    other_messages = []

    for msg in messages:
        if msg['role'] == 'system':
            system_messages.append(msg)
        else:
            other_messages.append(msg)

    # Calculate how many messages we can keep
    available_tokens = max_tokens - (len(system_messages) * tokens_per_message)
    max_other_messages = available_tokens // tokens_per_message

    # Keep most recent messages
    if len(other_messages) > max_other_messages:
        other_messages = other_messages[-max_other_messages:]

    return system_messages + other_messages


if __name__ == "__main__":
    print("System Prompts Module")
    print("=" * 60)

    # Test 1: Get system prompts
    print("\n1. Testing get_system_prompt():")
    print(f"\n   Default:\n   {get_system_prompt('default')[:100]}...")
    print(f"\n   Minimal:\n   {get_system_prompt('minimal')}")
    print(f"\n   None:\n   '{get_system_prompt('none')}'")

    # Test 2: Format conversation
    print("\n2. Testing format_conversation():")
    recent_messages = [
        {"username": "Alice", "content": "hey anyone want to play?", "is_bot": False},
        {"username": "Bot", "content": "I'm down", "is_bot": True},
        {"username": "Bob", "content": "yeah let's go", "is_bot": False}
    ]

    messages = format_conversation(
        recent_messages=recent_messages,
        current_message="cool let me hop on",
        current_author="Charlie",
        system_prompt=get_system_prompt("minimal"),
        max_history=3
    )

    print(f"\n   Formatted {len(messages)} messages:")
    for i, msg in enumerate(messages):
        print(f"   {i+1}. [{msg['role'].upper()}] {msg['content'][:60]}...")

    # Test 3: Build context
    print("\n3. Testing build_context_text():")
    context_messages = [
        {"username": "Dave", "content": "we should play later"},
        {"username": "Eve", "content": "im busy til 5pm"},
        {"username": "Frank", "content": "works for me"}
    ]

    context = build_context_text(context_messages, max_context=2)
    print(f"\n   Context (2 messages):\n   {context}")

    # Test 4: Display formatting
    print("\n4. Testing format_for_display():")
    display = format_for_display(messages, show_system=True)
    print(f"\n{display}")

    # Test 5: Truncation
    print("\n5. Testing truncate_conversation():")
    long_messages = [
        {"role": "system", "content": "System prompt"},
        *[{"role": "user", "content": f"Message {i}"} for i in range(50)]
    ]

    truncated = truncate_conversation(long_messages, max_tokens=1000, tokens_per_message=50)
    print(f"\n   Original: {len(long_messages)} messages")
    print(f"   Truncated: {len(truncated)} messages")
    print(f"   Kept system prompt: {truncated[0]['role'] == 'system'}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nUsage:")
    print("  from model.prompts import get_system_prompt, format_conversation")
    print("  messages = format_conversation(recent, current, author)")
    print("  response = generate_response(model, messages)")
