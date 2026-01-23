"""
Voice Chatbot Handler - Bridges STT -> LUMI -> TTS

Processes voice messages through the LUMI chatbot engine
while maintaining all therapeutic logic:
- Crisis detection
- RAG context retrieval
- Seven Paladins framework
- Citation generation
"""

import logging
import time
import uuid
import json
from typing import Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class VoiceChatbotHandler:
    """Handles voice messages through LUMI chatbot."""

    def __init__(self, chatbot):
        """
        Initialize with chatbot instance.

        Args:
            chatbot: TherapeuticChatbot instance
        """
        self.chatbot = chatbot

    def process_voice_message(
        self,
        db: Session,
        user_id: str,
        transcribed_text: str,
        language_code: str = "en-US",
        user_voice_preference: Optional[str] = None,
    ) -> Dict:
        """
        Process voice message through LUMI chatbot.

        This method:
        1. Validates transcribed text
        2. Calls chatbot.generate_response() (existing logic)
        3. Prepares response for TTS
        4. Stores voice interaction in database

        Args:
            db: Database session
            user_id: User identifier (WhatsApp number or web ID)
            transcribed_text: Text from STT service
            language_code: Language of original voice (e.g., "en-US")
            user_voice_preference: Preferred voice (e.g., "en-US-GuyNeural")

        Returns:
            Dict with:
            - bot_response: Text response from LUMI
            - is_crisis: Boolean flag
            - sources: List of citations
            - processing_time_seconds: Total latency
            - suggested_voice: Voice to use for TTS
            - success: Boolean
            - error_message: If error occurred
        """

        start_time = time.time()

        try:
            # Validate transcribed text
            if not transcribed_text or len(transcribed_text.strip()) == 0:
                return {
                    "success": False,
                    "error_message": "Transcribed text is empty",
                    "bot_response": None,
                    "sources": [],
                    "processing_time_seconds": time.time() - start_time,
                }

            logger.info(f"Processing voice message from {user_id}")
            logger.debug(f"Transcribed text: {transcribed_text[:100]}...")

            # Call existing chatbot logic
            # This handles ALL therapeutic features:
            # - Crisis detection
            # - RAG context retrieval
            # - Seven Paladins
            # - GPT-4 response generation
            response_dict = self.chatbot.generate_response(
                db=db,
                whatsapp_number=user_id,
                user_message=transcribed_text
            )

            bot_response = response_dict.get("response", "")
            is_crisis = response_dict.get("is_crisis", False)

            # Extract sources/citations from response
            # (included by chatbot in response_dict if using RAG)
            sources = self._extract_sources(bot_response)

            # Determine voice preference for TTS
            suggested_voice = user_voice_preference or "en-US-AriaNeural"

            # Store voice interaction in database
            self._store_voice_interaction(
                db=db,
                user_id=user_id,
                transcribed_text=transcribed_text,
                bot_response=bot_response,
                sources=sources,
                language_code=language_code,
                is_crisis=is_crisis,
            )

            processing_time = time.time() - start_time

            logger.info(f"Voice message processed in {processing_time:.2f}s")

            return {
                "success": True,
                "bot_response": bot_response,
                "is_crisis": is_crisis,
                "sources": sources,
                "suggested_voice": suggested_voice,
                "processing_time_seconds": processing_time,
                "error_message": None,
            }

        except Exception as e:
            logger.error(f"Error processing voice message: {e}", exc_info=True)

            return {
                "success": False,
                "bot_response": None,
                "is_crisis": False,
                "sources": [],
                "processing_time_seconds": time.time() - start_time,
                "error_message": str(e),
            }

    def _extract_sources(self, response_text: str) -> list:
        """
        Extract citations from response text.

        LUMI includes citations in format: "Text here. The Four Aces, p.25"

        Args:
            response_text: Response from LUMI

        Returns:
            List of source citations extracted from response
        """
        import re

        # Pattern: "Book Title, p.XX" or "Book Title, pp.XX-YY"
        pattern = r'([^.,]+),\s*p+\.?\s*(\d+(?:-\d+)?)'
        matches = re.findall(pattern, response_text)

        sources = [f"{book.strip()}, p.{pages}" for book, pages in matches]

        return sources if sources else []

    def _store_voice_interaction(
        self,
        db: Session,
        user_id: str,
        transcribed_text: str,
        bot_response: str,
        sources: list,
        language_code: str,
        is_crisis: bool,
    ) -> None:
        """
        Store voice interaction in database.

        Args:
            db: Database session
            user_id: User identifier
            transcribed_text: User's voice transcription
            bot_response: LUMI's response
            sources: Citations from response
            language_code: Language code
            is_crisis: Whether crisis detected
        """
        try:
            from database import VoiceInteraction

            voice_interaction = VoiceInteraction(
                id=uuid.uuid4(),
                user_id=user_id,
                user_transcription=transcribed_text,
                bot_response=bot_response,
                sources=sources,
                language_code=language_code,
                is_crisis=is_crisis,
                created_at=datetime.utcnow(),
                source_channel="voice",
            )

            db.add(voice_interaction)
            db.commit()

            logger.debug(f"Stored voice interaction for user {user_id}")

        except Exception as e:
            logger.warning(f"Could not store voice interaction: {e}")
            # Don't fail the entire request if DB storage fails
