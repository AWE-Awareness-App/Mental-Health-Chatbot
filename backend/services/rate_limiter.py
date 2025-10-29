"""
Basic rate limiting for WhatsApp webhook.
Prevents spam and abuse by limiting messages per user.
"""

import time
from typing import Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_messages: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_messages: Maximum messages allowed per window
            window_seconds: Time window in seconds
        """
        self.max_messages = max_messages
        self.window_seconds = window_seconds
        self.user_requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> bool:
        """
        Check if user is allowed to send message.
        
        Args:
            user_id: User identifier (e.g., WhatsApp number)
        
        Returns:
            True if allowed, False if rate limited
        """
        current_time = time.time()
        
        # Get user's request history
        user_history = self.user_requests[user_id]
        
        # Remove old requests outside the window
        user_history = [t for t in user_history if current_time - t < self.window_seconds]
        self.user_requests[user_id] = user_history
        
        # Check if user exceeded rate limit
        if len(user_history) >= self.max_messages:
            logger.warning(f"⚠️ Rate limit exceeded for user {user_id}")
            return False
        
        # Add current request
        user_history.append(current_time)
        return True
    
    def get_retry_after(self, user_id: str) -> int:
        """
        Get seconds until user can send another message.
        
        Args:
            user_id: User identifier
        
        Returns:
            Seconds to wait, or 0 if can send now
        """
        current_time = time.time()
        user_history = self.user_requests[user_id]
        
        if not user_history or len(user_history) < self.max_messages:
            return 0
        
        # Calculate when the oldest request will expire
        oldest_request = min(user_history)
        retry_after = int(self.window_seconds - (current_time - oldest_request))
        
        return max(0, retry_after)
    
    def reset_user(self, user_id: str):
        """Reset rate limit for a specific user."""
        if user_id in self.user_requests:
            del self.user_requests[user_id]
            logger.info(f"✅ Rate limit reset for user {user_id}")


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_messages=10, window_seconds=60)
        logger.info("✅ Rate limiter initialized (10 messages per 60 seconds)")
    return _rate_limiter
