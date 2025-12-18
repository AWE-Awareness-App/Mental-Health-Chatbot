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
- LUMI identity and self-identification
- Seven Paladins of Positivity framework
- Professional referral suggestions

Enhancements:
- Improved error handling
- More stable API fallback system
- Handles missing DB columns safely (helpful during schema migration)
- Better logging for debugging & observability
- LUMI identity (AI, not therapist, not human)
- Seven Paladins for reframing negative situations
- Professional escalation for serious mental health topics
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
    get_conversation_history,
    aichatusers
)

logger = logging.getLogger(__name__)

# =========================================================================
# LUMI IDENTITY & SEVEN PALADINS CONSTANTS
# =========================================================================

# LUMI Identity Introduction
LUMI_INTRODUCTION = (
    "I am LUMI, your digital wellness and happiness coach here to guide you "
    "to find balance, better relationships, and more joy even in our digital age. "
    "\n\n"
    "I leverage insights from habit formation and various frameworks like the Four Aces, "
    "the 7Cs, and 8Ps to help you build a life you love. "
    "\n\n"
    "However, I'm an AI assistant — not a therapist or a doctor. If you're facing "
    "serious mental health challenges, trauma, or crises, I'd recommend connecting with "
    "an AWE human coach or specialist who can provide personalized professional support. "
    "\n\n"
    "How can I help you on your wellness journey today?"
)

# Seven Paladins of Positivity (From "Beyond Happy")
SEVEN_PALADINS = {
    "perspective": (
        "Let me offer a different perspective. Sometimes our first instinct is to see "
        "the worst-case scenario. But what if we zoomed out? "
        "\n\n"
        "Ask yourself: On a scale of 1-10, how much will this situation matter in: "
        "1 week? 1 month? 1 year? "
        "\n\n"
        "This isn't about dismissing your feelings — it's about seeing the fuller picture. "
        "What would that bigger view show?"
    ),

    "agency": (
        "I hear that you feel stuck. But here's what I know: you always have more "
        "power than you think. Even in tough situations, there's usually something "
        "you can influence or control. "
        "\n\n"
        "What's ONE small thing — no matter how tiny — that you could do differently "
        "in the next 24 hours? Start with the smallest possible step. "
        "\n\n"
        "Remember: You don't need to fix everything at once. You just need one move "
        "in a better direction. What could that be?"
    ),

    "compassion": (
        "I'm noticing you're being really hard on yourself. Let me pause here and ask: "
        "If a friend came to you with this exact situation, what would you say to them? "
        "\n\n"
        "Usually, we're much kinder to others than we are to ourselves. But you deserve "
        "that same kindness and understanding. What mistakes haven't you made? We all "
        "make choices we later question — that's called being human. "
        "\n\n"
        "What would self-compassion look like for you right now?"
    ),

    "gratitude": (
        "I know things feel heavy right now. But let's try something different. "
        "\n\n"
        "Even on rough days, there's usually something — no matter how small — that "
        "went okay or that we can appreciate. "
        "\n\n"
        "Can you name: "
        "1) One thing (even tiny) that worked today? "
        "2) One person who was kind to you recently? "
        "3) One thing your body did right (you're breathing, thinking, reaching out)? "
        "\n\n"
        "This isn't about ignoring the hard stuff. It's about seeing the fuller picture. "
        "What comes up for you?"
    ),

    "curiosity": (
        "I'm curious about something. Instead of stopping at what they did wrong, "
        "let's dig deeper. "
        "\n\n"
        "Ask yourself: "
        "- What might they have been feeling or struggling with? "
        "- What was their story behind the action? "
        "- What would understanding their perspective add? "
        "\n\n"
        "This doesn't mean they weren't wrong. It means understanding why things happened — "
        "which often reduces pain and opens pathways to solutions. "
        "\n\n"
        "What do you think was really going on for them?"
    ),

    "growth": (
        "Okay, that didn't go as you hoped. But here's the truth: every struggle, "
        "setback, and 'failure' is actually information. "
        "\n\n"
        "What's one thing this situation taught you? What did you learn about: "
        "- What matters to you? "
        "- What you'd do differently next time? "
        "- What you're actually capable of handling? "
        "\n\n"
        "This is the difference between a dead-end and a detour. Which one is this? "
        "What's the growth opportunity hiding here?"
    ),

    "connection": (
        "One of the deepest human needs is connection. If you're feeling alone in this, "
        "I want you to know: you're not. Millions of people have felt exactly what you're feeling. "
        "\n\n"
        "Reaching out — even to me, a digital coach — was a brave act. "
        "\n\n"
        "Consider: "
        "- Who in your life could you be honest with about what you're going through? "
        "- What community or group shares your interests or challenges? "
        "- How could you connect with one person this week? "
        "\n\n"
        "Connection isn't about fixing things. It's about not facing things alone. "
        "Who could you reach out to?"
    )
}

# Trigger Keywords for Each Paladin
PALADIN_TRIGGERS = {
    "perspective": [
        "worst", "catastrophic", "ruined", "can't handle", "never get better",
        "everything is bad", "all ruined", "hopeless", "end it all", "unbearable"
    ],

    "agency": [
        "nothing i can do", "nothing i can", "helpless", "powerless", "stuck",
        "no choice", "no control", "what's the point", "fate", "destiny"
    ],

    "compassion": [
        "my fault", "my bad", "i'm stupid", "i'm dumb", "i always mess up",
        "i never", "i should have", "i'm so", "stupid", "idiot", "failure",
        "blame myself", "my mistake"
    ],

    "gratitude": [
        "my day", "my week", "my month", "is ruined", "was terrible", "nothing good",
        "everything sucks", "all bad", "complete disaster", "the worst", "bad day"
    ],

    "curiosity": [
        "they don't", "they are so", "they always", "they never", "everyone against",
        "nobody cares", "selfish", "uncaring", "always", "never"
    ],

    "growth": [
        "i failed", "failure", "didn't work", "made a mistake", "messed up",
        "went wrong", "hopeless", "useless", "pointless", "wasted"
    ],

    "connection": [
        "alone", "lonely", "nobody", "no one", "isolated", "no support",
        "nobody understands", "nobody cares", "i'm by myself", "have no one"
    ]
}

# Professional Referral
PROFESSIONAL_REFERRAL = (
    "\n\n"
    "I can tell this is really important to you, and I want to be honest: "
    "while I can offer frameworks and digital support, what you're describing "
    "might benefit from working with an AWE human coach or specialist. "
    "\n\n"
    "They can: "
    "✓ Provide personalized, ongoing support "
    "✓ Help you process deeper patterns "
    "✓ Offer specialized expertise for your situation "
    "✓ Build a relationship over time "
    "\n\n"
    "Would you be open to exploring that option? "
    "Or would you like to continue with me while you consider it? "
    "\n\n"
    "Remember: seeking professional help isn't a weakness — it's wisdom."
)

# Referral Triggers (When to suggest professional help)
REFERRAL_KEYWORDS = [
    "therapist", "therapy", "depression", "depressed", "anxiety", "anxious",
    "ptsd", "trauma", "grief", "grieving", "addiction", "addicted", "eating disorder",
    "suicidal", "self harm", "self-harm", "professional help", "need help", "serious"
]


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
    # LUMI IDENTITY
    # -------------------------------------------------------------------------

    def is_identity_question(self, message: str) -> bool:
        """Check if user is asking about LUMI's identity."""
        identity_keywords = [
            "who are you", "what are you", "are you a therapist", "are you human",
            "are you real", "tell me about you", "what's your name", "who's lumi",
            "what is lumi", "introduce yourself"
        ]
        return any(keyword in message.lower() for keyword in identity_keywords)

    # -------------------------------------------------------------------------
    # SEVEN PALADINS DETECTION & APPLICATION
    # -------------------------------------------------------------------------

    def detect_paladin_needed(self, message: str) -> Optional[str]:
        """
        Detect if conversation needs one of the Seven Paladins.
        Returns: paladin_name or None
        """
        text = message.lower()

        for paladin, triggers in PALADIN_TRIGGERS.items():
            if any(trigger in text for trigger in triggers):
                logger.info(f"🌟 Paladin detected: {paladin}")
                return paladin

        return None

    def get_paladin_response(self, paladin_name: str) -> str:
        """Get the response text for a specific Paladin."""
        return SEVEN_PALADINS.get(paladin_name, "")

    # -------------------------------------------------------------------------
    # PROFESSIONAL REFERRAL DETECTION
    # -------------------------------------------------------------------------

    def should_suggest_referral(self, message: str, conversation_turn: int = 0) -> bool:
        """
        Determine if user should be offered professional referral.

        Triggers:
        - Explicitly mentions need for professional help
        - Serious mental health keywords
        - Same topic repeated 5+ times
        """
        text = message.lower()

        # Explicit referral triggers
        if any(keyword in text for keyword in REFERRAL_KEYWORDS):
            return True

        # If conversation has gone 5+ turns on serious topic, suggest escalation
        if conversation_turn >= 5:
            serious_keywords = ["therapy", "therapist", "professional", "depression", "anxiety", "trauma"]
            if any(keyword in text for keyword in serious_keywords):
                return True

        return False

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
              "user_id": <string or None>,
              "conversation_turn": <int>
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
                    "user_id": str(user.id),
                    "conversation_turn": 0
                }

            # -------------------------
            # 3. LUMI IDENTITY CHECK
            # -------------------------
            if self.is_identity_question(user_message):
                logger.info("💬 User asking about LUMI identity")
                try:
                    conversation = get_active_conversation(db, user.id)
                except Exception:
                    conversation = None

                # Save identity question & response
                if conversation:
                    try:
                        user_embed = self.rag.create_embedding(user_message) if self.rag else None
                        identity_embed = self.rag.create_embedding(LUMI_INTRODUCTION) if self.rag else None

                        save_message(db, conversation.id, user.id, "user", user_message, user_embed)
                        save_message(db, conversation.id, user.id, "assistant", LUMI_INTRODUCTION, identity_embed)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not save identity exchange: {e}")

                elapsed = time.time() - start_time
                logger.info(f"✓ Identity response in {elapsed:.2f}s")
                return {
                    "response": LUMI_INTRODUCTION,
                    "is_crisis": False,
                    "user_id": str(user.id),
                    "conversation_turn": 1
                }

            # -------------------------
            # 4. SEVEN PALADINS CHECK
            # -------------------------
            paladin_name = self.detect_paladin_needed(user_message)
            paladin_context = ""

            if paladin_name:
                paladin_context = f"\n\nApproach: Use the {paladin_name.upper()} Paladin to help reframe this situation positively."
                logger.info(f"🌟 Applying {paladin_name} Paladin")

            # -------------------------
            # 5. Normal Conversation Flow
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

            # Build prompt with Paladin context
            try:
                if self.rag:
                    prompt = self.rag.build_prompt_with_context(
                        user_message, relevant_contexts, conversation_history
                    )
                else:
                    prompt = f"You are LUMI, a compassionate digital wellness coach. User message: {user_message}"

                # Add Paladin context if needed
                if paladin_context:
                    prompt += paladin_context
            except Exception:
                prompt = f"You are LUMI, a compassionate digital wellness coach. User message: {user_message}"
                if paladin_context:
                    prompt += paladin_context

            # -------------------------
            # 6. GPT-4 Response (Retry Logic)
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

            # Fallback if GPT-4 fails
            if not bot_response:
                # Use Paladin response if available
                if paladin_name:
                    bot_response = self.get_paladin_response(paladin_name)
                else:
                    bot_response = self._get_fallback_response(user_message)

            # -------------------------
            # 7. Check for Professional Referral
            # -------------------------
            conversation_turn = len(conversation_history) // 2 if conversation_history else 0

            if self.should_suggest_referral(user_message, conversation_turn):
                bot_response += PROFESSIONAL_REFERRAL
                logger.info(f"🔗 Suggested professional referral")

            # -------------------------
            # 8. Save Messages
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
                "user_id": str(user.id),
                "conversation_turn": conversation_turn
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
                "user_id": None,
                "conversation_turn": 0
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
                "What's one small thing within your control today? 🌿"
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
