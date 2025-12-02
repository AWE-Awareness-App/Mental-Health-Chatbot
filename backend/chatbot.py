"""
Therapeutic Chatbot Logic (MVP Version)
--------------------------------------

This module implements a therapeutic chatbot with OpenAI GPT-4 integration.  
It supports:

- Crisis message detection
- GPT-4 response generation
- RAG-enhanced context retrieval
- Conversation continuity
- Robust error handling with graceful degradation
- Fallback responses when APIs fail
- Logging, metrics, and database failure-safe behavior

Enhancements:
- Improved error handling
- More stable API fallback system
- Handles missing DB columns safely (helpful during schema migration)
- Better logging for debugging & observability
"""

import os
import re
import time
import logging
from typing import List, Dict, Optional

from openai import OpenAI, OpenAIError, RateLimitError, APIError
from sqlalchemy.orm import Session

from rag_system_v2 import TherapeuticRAG
from database import (
    get_or_create_user,
    get_active_conversation,
    save_message,
    get_conversation_history
)

logger = logging.getLogger(__name__)


class TherapeuticChatbot:
    """Main chatbot class for handling therapeutic conversations."""

    # Keywords that indicate a psychological crisis
    CRISIS_KEYWORDS = [
        "suicide", "suicidal", "kill myself", "end my life", "want to die",
        "self-harm", "hurt myself", "cutting", "overdose", "no reason to live",
        "better off dead", "hopeless", "can't go on"
    ]

    # Crisis response template
    CRISIS_RESPONSE = (
        "I hear that you're going through an extremely difficult time, and I'm concerned "
        "about your safety. Your life matters, and there are people who want to help you right now.\n\n"
        "**Please reach out for immediate support:**\n\n"
        "🆘 National Suicide Prevention Lifeline: 988 (call or text)\n"
        "💬 Crisis Text Line: Text HOME to 741741\n"
        "🌐 SAMHSA National Helpline: 1-800-662-4357\n\n"
        "These services are free, confidential, and available 24/7. Trained counselors are ready to listen.\n\n"
        "If you're in immediate danger, please call 911 or visit the nearest emergency room.\n\n"
        "I'm here to support you with digital wellness, but professional crisis counselors "
        "are best equipped to help with these intense feelings. Would you like to share what led "
        "you to reach out today?"
    )

    def __init__(self, openai_api_key: str, rag_system: Optional[TherapeuticRAG] = None):
        """
        Initialize chatbot with OpenAI client and optional RAG system.

        Args:
            openai_api_key: OpenAI API key
            rag_system: Optional RAG instance initialized in server.py
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.rag = rag_system
        self.model = "gpt-4-turbo-preview"

    # -------------------------------------------------------------------------
    # Crisis Detection
    # -------------------------------------------------------------------------

    def detect_crisis(self, message: str) -> bool:
        """Return True if the message contains crisis-related keywords."""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.CRISIS_KEYWORDS)

    # -------------------------------------------------------------------------
    # User Handling (Safe Mode)
    # -------------------------------------------------------------------------

    def _get_or_create_user_safe(self, db: Session, whatsapp_number: str):
        """
        Safely retrieve or create a user object.
        Avoids breaking when DB columns are missing.

        Returns:
            User object, or a fallback MinimalUser object if DB query fails.
        """
        try:
            user = get_or_create_user(db, whatsapp_number)
            logger.info(f"✓ User retrieved/created: {whatsapp_number}")
            return user

        except Exception as e:
            logger.warning(f"⚠️ Could not get/create user: {e}")

            # Fallback minimal user representation
            class MinimalUser:
                def __init__(self, phone):
                    self.id = hash(phone)
                    self.whatsapp_number = phone
                    self.crisis_flag = False

            return MinimalUser(whatsapp_number)

    # -------------------------------------------------------------------------
    # Response Generation
    # -------------------------------------------------------------------------

    def generate_response(self, db: Session, whatsapp_number: str, user_message: str) -> Dict:
        """
        Generate a therapeutic response using GPT-4 + RAG (if available).

        Returns:
            {
              "response": <string>,
              "is_crisis": <bool>,
              "user_id": <string or None>
            }
        """
        start_time = time.time()

        try:
            # -------------------------
            # 1. Get User (Safe)
            # -------------------------
            user = self._get_or_create_user_safe(db, whatsapp_number)

            # -------------------------
            # 2. Crisis Detection
            # -------------------------
            is_crisis = self.detect_crisis(user_message)

            if is_crisis:
                logger.warning(f"⚠️ Crisis content detected from {whatsapp_number}")

                # Attempt to set crisis flag
                try:
                    if hasattr(user, "crisis_flag"):
                        user.crisis_flag = True
                        db.commit()
                except Exception as e:
                    logger.warning(f"⚠️ Could not save crisis flag: {e}")

                # Save crisis message
                try:
                    conversation = get_active_conversation(db, user.id)
                    crisis_embedding = None

                    if self.rag:
                        try:
                            crisis_embedding = self.rag.create_embedding(self.CRISIS_RESPONSE)
                        except Exception:
                            crisis_embedding = None

                    save_message(
                        db, conversation.id, user.id, "assistant",
                        self.CRISIS_RESPONSE, crisis_embedding
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Could not save crisis message: {e}")

                return {
                    "response": self.CRISIS_RESPONSE,
                    "is_crisis": True,
                    "user_id": str(user.id)
                }

            # -------------------------
            # 3. Normal Conversation Flow
            # -------------------------

            try:
                conversation = get_active_conversation(db, user.id)
            except Exception as e:
                logger.warning(f"⚠️ Could not get conversation: {e}")
                conversation = None

            # Retrieve history
            try:
                if conversation:
                    history_messages = get_conversation_history(db, conversation.id, limit=10)
                    conversation_history = [
                        {"role": m.role, "content": m.content}
                        for m in history_messages
                    ]
                else:
                    conversation_history = []

            except Exception as e:
                logger.warning(f"⚠️ Error retrieving history: {e}")
                conversation_history = []

            # Retrieve RAG context
            relevant_contexts = []
            if self.rag:
                try:
                    relevant_contexts = self.rag.retrieve_relevant_context(
                        db, user_message, k=5
                    )
                except Exception as e:
                    logger.error(f"❌ Error retrieving RAG context: {e}")

            # Build prompt
            try:
                if self.rag:
                    prompt = self.rag.build_prompt_with_context(
                        user_message, relevant_contexts, conversation_history
                    )
                else:
                    prompt = f"You are a compassionate digital wellness therapist. User message: {user_message}"
            except Exception:
                prompt = f"You are a compassionate digital wellness therapist. User message: {user_message}"

            # -------------------------
            # 4. GPT-4 Response (Retry Logic)
            # -------------------------

            bot_response = None

            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=0.7,
                        max_tokens=300,
                        presence_penalty=0.6,
                        frequency_penalty=0.3
                    )

                    bot_response = response.choices[0].message.content.strip()
                    break

                except RateLimitError:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"⚠️ Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)

                except APIError as e:
                    logger.error(f"❌ OpenAI API error: {e}")
                    if attempt < 2:
                        time.sleep(1)

                except Exception as e:
                    logger.error(f"❌ GPT-4 call failed: {e}")
                    break

            # Fallback response if GPT-4 fails
            if not bot_response:
                bot_response = self._get_fallback_response(user_message)

            # -------------------------
            # 5. Save Messages
            # -------------------------
            try:
                if conversation:
                    # Save user msg
                    try:
                        user_embed = self.rag.create_embedding(user_message) if self.rag else None
                    except Exception:
                        user_embed = None

                    save_message(
                        db, conversation.id, user.id,
                        "user", user_message, user_embed
                    )

                    # Save bot msg
                    try:
                        bot_embed = self.rag.create_embedding(bot_response) if self.rag else None
                    except Exception:
                        bot_embed = None

                    save_message(
                        db, conversation.id, user.id,
                        "assistant", bot_response, bot_embed
                    )
            except Exception as e:
                logger.warning(f"⚠️ Could not save messages: {e}")

            # Completed
            elapsed = time.time() - start_time
            logger.info(f"✓ Response generated in {elapsed:.2f}s")

            return {
                "response": bot_response,
                "is_crisis": False,
                "user_id": str(user.id)
            }

        except Exception as e:
            logger.error(f"❌ Critical error: {e}", exc_info=True)

            fallback = (
                "I'm sorry — I'm having temporary technical issues.\n\n"
                "If you're experiencing a crisis, please contact:\n"
                "- 988 Suicide & Crisis Lifeline\n"
                "- Text HOME to 741741\n\n"
                "Please try again in a moment."
            )

            return {
                "response": fallback,
                "is_crisis": False,
                "user_id": None
            }

    # -------------------------------------------------------------------------
    # Fallback Responses
    # -------------------------------------------------------------------------

    def _get_fallback_response(self, user_message: str) -> str:
        """
        Generate contextual fallback if GPT-4 is unavailable.
        """

        text = user_message.lower()

        # Screen time concerns
        if any(word in text for word in ["screen", "phone", "instagram", "tiktok", "social media"]):
            return (
                "I understand you're concerned about screen time. Here's a quick technique:\n\n"
                "Try the **Four Aces** approach — start with *Awareness*. Notice your phone usage "
                "patterns without judgment. What usually triggers you to reach for your phone? 📱\n\n"
                "Can you identify one trigger right now?"
            )

        # Stress / anxiety
        if any(word in text for word in ["stress", "anxiety", "worried", "anxious"]):
            return (
                "It sounds like you're feeling stressed. Try this simple grounding technique:\n\n"
                "Take **three slow breaths** and focus on what you *can* control right now.\n"
                "What’s one small thing within your control today? 🌿"
            )

        # Habit/addiction
        if any(word in text for word in ["addicted", "habit", "can't stop"]):
            return (
                "Breaking habits is tough. Start with **Acceptance** — acknowledge the habit "
                "without judging yourself. Then ask: what's one *tiny* step you can take today?\n\n"
                "Remember: progress over perfection. 💪"
            )

        # General fallback
        return (
            "I'm here to support you. I'm having temporary technical issues, but your feelings are valid.\n\n"
            "Take a moment for yourself — maybe step outside or take a few slow breaths.\n"
            "What's one small act of self-care you can do right now? 🌟"
        )

    # -------------------------------------------------------------------------
    # WhatsApp Formatting
    # -------------------------------------------------------------------------

    def format_whatsapp_message(self, text: str) -> str:
        """Clean and standardize message formatting for WhatsApp."""
        formatted = text.strip()
        formatted = re.sub(r"\n\n\n+", "\n\n", formatted)
        return formatted


# -------------------------------------------------------------------------
# Singleton Instance
# -------------------------------------------------------------------------

_chatbot_instance: Optional[TherapeuticChatbot] = None


def get_chatbot() -> TherapeuticChatbot:
    """Return singleton chatbot instance."""
    global _chatbot_instance

    if _chatbot_instance is None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY is not set")

        _chatbot_instance = TherapeuticChatbot(api_key)
        logger.info("Chatbot instance created")

    return _chatbot_instance


# -------------------------------------------------------------------------
# Standalone Test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        chatbot = get_chatbot()
        print("Chatbot initialized successfully!")
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
