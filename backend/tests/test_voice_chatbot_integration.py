"""
Integration Tests: Voice + LUMI Chatbot

Tests complete voice flow:
STT -> LUMI Processing -> TTS
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from services.voice_chatbot_handler import VoiceChatbotHandler


class TestVoiceChatbotIntegration:
    """Test voice integration with LUMI chatbot."""

    @pytest.fixture
    def mock_chatbot(self):
        """Create mock chatbot."""
        mock = Mock()
        mock.generate_response.return_value = {
            "response": "I understand you're feeling stressed. Try the Four Aces approach: start with Awareness. The Four Aces, p.32",
            "is_crisis": False,
            "user_id": "test-user-123",
            "conversation_turn": 1,
        }
        return mock

    @pytest.fixture
    def handler(self, mock_chatbot):
        """Create voice handler with mock chatbot."""
        return VoiceChatbotHandler(mock_chatbot)

    def test_process_voice_message_success(self, handler, mock_chatbot):
        """Test successful voice message processing."""

        mock_db = Mock()

        result = handler.process_voice_message(
            db=mock_db,
            user_id="whatsapp_1234567890",
            transcribed_text="I'm feeling stressed about work.",
            language_code="en-US",
            user_voice_preference="en-US-AriaNeural",
        )

        assert result["success"] is True
        assert result["bot_response"] is not None
        assert result["is_crisis"] is False
        assert result["sources"] == ["The Four Aces, p.32"]
        assert result["suggested_voice"] == "en-US-AriaNeural"
        assert result["processing_time_seconds"] > 0

        # Verify chatbot was called
        mock_chatbot.generate_response.assert_called_once()

    def test_crisis_detection_in_voice(self, handler, mock_chatbot):
        """Test crisis detection in voice input."""

        # Mock crisis response
        mock_chatbot.generate_response.return_value = {
            "response": "I hear you're in crisis. Please contact 988...",
            "is_crisis": True,
            "user_id": "test-user",
            "conversation_turn": 1,
        }

        mock_db = Mock()

        result = handler.process_voice_message(
            db=mock_db,
            user_id="whatsapp_9876543210",
            transcribed_text="I want to end my life",
            language_code="en-US",
        )

        assert result["success"] is True
        assert result["is_crisis"] is True

    def test_empty_transcription_handling(self, handler):
        """Test handling of empty transcription."""

        mock_db = Mock()

        result = handler.process_voice_message(
            db=mock_db,
            user_id="test-user",
            transcribed_text="",  # Empty
            language_code="en-US",
        )

        assert result["success"] is False
        assert "empty" in result["error_message"].lower()

    def test_whitespace_only_transcription(self, handler):
        """Test handling of whitespace-only transcription."""

        mock_db = Mock()

        result = handler.process_voice_message(
            db=mock_db,
            user_id="test-user",
            transcribed_text="   ",  # Whitespace only
            language_code="en-US",
        )

        assert result["success"] is False
        assert "empty" in result["error_message"].lower()

    def test_voice_with_rag_context(self, handler, mock_chatbot):
        """Test voice message with RAG context retrieval."""

        # Mock response with multiple citations
        mock_chatbot.generate_response.return_value = {
            "response": "The Four Aces framework (Awareness, Acceptance, Appreciation, Awe) can help. The Four Aces, p.25. Additionally, the 7Cs framework... Beyond Happy, p.45",
            "is_crisis": False,
            "user_id": "test-user",
            "conversation_turn": 2,
        }

        mock_db = Mock()

        result = handler.process_voice_message(
            db=mock_db,
            user_id="whatsapp_1234567890",
            transcribed_text="Tell me about happiness frameworks",
            language_code="en-US",
        )

        assert result["success"] is True
        # Should extract multiple sources
        assert len(result["sources"]) >= 2
        assert any("Four Aces" in source for source in result["sources"])

    def test_language_preference_handling(self):
        """Test different language inputs."""

        mock_chatbot = Mock()
        handler_es = VoiceChatbotHandler(mock_chatbot)

        mock_chatbot.generate_response.return_value = {
            "response": "Entiendo que te sientes estresado...",
            "is_crisis": False,
            "user_id": "test-user",
            "conversation_turn": 1,
        }

        mock_db = Mock()

        result = handler_es.process_voice_message(
            db=mock_db,
            user_id="whatsapp_user",
            transcribed_text="Estoy estresado",
            language_code="es-ES",
            user_voice_preference="es-ES-ElviraNeural",
        )

        assert result["success"] is True
        assert result["suggested_voice"] == "es-ES-ElviraNeural"

    def test_default_voice_when_no_preference(self, handler, mock_chatbot):
        """Test that default voice is used when no preference is provided."""

        mock_db = Mock()

        result = handler.process_voice_message(
            db=mock_db,
            user_id="test-user",
            transcribed_text="Hello there",
            language_code="en-US",
            user_voice_preference=None,  # No preference
        )

        assert result["success"] is True
        assert result["suggested_voice"] == "en-US-AriaNeural"  # Default

    def test_chatbot_error_handling(self, handler, mock_chatbot):
        """Test handling when chatbot raises an exception."""

        mock_chatbot.generate_response.side_effect = Exception("OpenAI API error")
        mock_db = Mock()

        result = handler.process_voice_message(
            db=mock_db,
            user_id="test-user",
            transcribed_text="Hello",
            language_code="en-US",
        )

        assert result["success"] is False
        assert result["error_message"] is not None
        assert "OpenAI API error" in result["error_message"]

    def test_source_extraction_single_citation(self, handler):
        """Test extracting a single citation from response."""

        sources = handler._extract_sources(
            "This is a test response. The Four Aces, p.32"
        )

        assert len(sources) == 1
        assert "Four Aces" in sources[0]
        assert "32" in sources[0]

    def test_source_extraction_multiple_citations(self, handler):
        """Test extracting multiple citations from response."""

        sources = handler._extract_sources(
            "First reference: The Four Aces, p.25. Second: Beyond Happy, p.100-105"
        )

        assert len(sources) == 2

    def test_source_extraction_no_citations(self, handler):
        """Test response with no citations returns empty list."""

        sources = handler._extract_sources(
            "This is a response without any book citations."
        )

        assert sources == []


class TestVoiceChatbotHandlerInit:
    """Test VoiceChatbotHandler initialization."""

    def test_init_with_chatbot(self):
        """Test initialization with chatbot instance."""
        mock_chatbot = Mock()
        handler = VoiceChatbotHandler(mock_chatbot)

        assert handler.chatbot == mock_chatbot

    def test_init_stores_chatbot_reference(self):
        """Test that chatbot reference is stored correctly."""
        mock_chatbot = Mock()
        mock_chatbot.some_method = Mock(return_value="test")

        handler = VoiceChatbotHandler(mock_chatbot)

        assert handler.chatbot.some_method() == "test"


class TestVoiceInteractionStorage:
    """Test voice interaction database storage."""

    def test_store_voice_interaction_success(self):
        """Test successful storage of voice interaction."""
        mock_chatbot = Mock()
        mock_chatbot.generate_response.return_value = {
            "response": "Test response",
            "is_crisis": False,
            "user_id": "test-user",
            "conversation_turn": 1,
        }

        handler = VoiceChatbotHandler(mock_chatbot)

        # Mock database session
        mock_db = Mock()
        mock_db.add = Mock()
        mock_db.commit = Mock()

        result = handler.process_voice_message(
            db=mock_db,
            user_id="test-user",
            transcribed_text="Test message",
            language_code="en-US",
        )

        # Should have succeeded
        assert result["success"] is True
        assert result["bot_response"] == "Test response"

    def test_store_voice_interaction_db_failure_non_fatal(self):
        """Test that DB storage failure doesn't fail the entire request."""
        mock_chatbot = Mock()
        mock_chatbot.generate_response.return_value = {
            "response": "Test response",
            "is_crisis": False,
            "user_id": "test-user",
            "conversation_turn": 1,
        }

        handler = VoiceChatbotHandler(mock_chatbot)
        mock_db = Mock()
        mock_db.add.side_effect = Exception("DB error")

        # Should still succeed even if DB storage fails
        result = handler.process_voice_message(
            db=mock_db,
            user_id="test-user",
            transcribed_text="Test message",
            language_code="en-US",
        )

        # Request should still succeed - DB failure is non-fatal
        assert result["success"] is True
        assert result["bot_response"] == "Test response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
