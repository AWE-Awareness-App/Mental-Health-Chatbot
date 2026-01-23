"""
Service modules for AWE Mental Health Chatbot.

This package contains voice integration services:
- speech_to_text: Azure Speech Services STT integration
- text_to_speech: Azure Speech Services TTS integration

Author: Emergent
Version: 1.0.0
"""

from .speech_to_text import SpeechToTextService, get_stt_service, TranscriptionResult
from .text_to_speech import TextToSpeechService, get_tts_service, SynthesisResult, VoiceInfo

__all__ = [
    "SpeechToTextService",
    "get_stt_service",
    "TranscriptionResult",
    "TextToSpeechService",
    "get_tts_service",
    "SynthesisResult",
    "VoiceInfo",
]
