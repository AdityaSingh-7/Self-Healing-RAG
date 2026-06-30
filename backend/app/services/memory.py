"""
services/memory.py — Conversation Memory

WHAT THIS DOES:
Manages conversation history so the LLM remembers previous messages.
Without this, every query is independent — the system forgets immediately.

THE PROBLEM WITHOUT MEMORY:
    User: "What's our PTO policy?"
    AI: "20 days per year (handbook, page 12)"
    User: "Can I carry it over?"
    AI: "I don't have enough context." ← Forgot what "it" refers to!

WITH MEMORY:
    We inject the last N messages into the prompt, so the LLM knows
    "it" = "PTO days" from the previous exchange.

DESIGN DECISIONS:
- Sliding window (last N messages) — simple, bounded memory
- Why not infinite history? LLM context windows have limits (128K for Llama 3)
  and more context = slower + more expensive
- We keep the last 10 message pairs (user + assistant = 20 messages max)
- Oldest messages drop off naturally

STORAGE:
For now, we store in memory (dict). In production, you'd use Redis or a database
so conversations survive server restarts. But for a portfolio project, in-memory is fine.
"""


class ConversationMemory:
    """
    In-memory conversation store.

    Each user has their own conversation history, keyed by user_id.
    History is a list of {"role": "user"|"assistant", "content": "..."} dicts.

    Usage:
        memory = ConversationMemory()
        memory.add_message("user123", "user", "What is PTO?")
        memory.add_message("user123", "assistant", "PTO is 20 days...")
        history = memory.get_history("user123")
    """

    # Maximum number of message pairs to keep (10 pairs = 20 messages)
    MAX_PAIRS = 10

    def __init__(self):
        """
        Initialize the memory store.

        _conversations is a dict:
        {
            "user_123": [
                {"role": "user", "content": "What is PTO?"},
                {"role": "assistant", "content": "PTO is 20 days..."},
                ...
            ],
            "user_456": [...],
        }
        """
        self._conversations: dict[str, list[dict]] = {}

    def get_history(self, user_id: str) -> list[dict]:
        """
        Get the conversation history for a user.

        Returns the last MAX_PAIRS * 2 messages (user + assistant pairs).
        Returns empty list if no history exists.
        """
        return self._conversations.get(user_id, [])

    def add_message(self, user_id: str, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Parameters:
        -----------
        user_id : str
            Which user's conversation this belongs to
        role : str
            "user" or "assistant"
        content : str
            The message text
        """
        # Create history list if this is the user's first message
        if user_id not in self._conversations:
            self._conversations[user_id] = []

        # Append the new message
        self._conversations[user_id].append({
            "role": role,
            "content": content,
        })

        # Trim to keep only the last MAX_PAIRS * 2 messages
        # (10 pairs = 20 messages max)
        max_messages = self.MAX_PAIRS * 2
        if len(self._conversations[user_id]) > max_messages:
            # Keep only the tail (newest messages)
            self._conversations[user_id] = self._conversations[user_id][-max_messages:]

    def clear(self, user_id: str) -> None:
        """
        Clear all conversation history for a user.
        Call this when the user starts a "new conversation."
        """
        self._conversations.pop(user_id, None)

    def get_summary_stats(self, user_id: str) -> dict:
        """
        Get stats about a user's conversation (for debugging/display).
        """
        history = self.get_history(user_id)
        return {
            "message_count": len(history),
            "user_messages": sum(1 for m in history if m["role"] == "user"),
            "assistant_messages": sum(1 for m in history if m["role"] == "assistant"),
        }


# Global singleton instance — import this in other files
# We use a singleton so all endpoints share the same memory state
conversation_memory = ConversationMemory()
