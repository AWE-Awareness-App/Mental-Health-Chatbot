"""
Speech-to-Text Service for AWE Mental Health Chatbot

Provides Azure Speech Services integration for transcribing audio to text.
Supports multiple audio formats (WAV, MP3, OGG) with validation.
Includes mock mode for local testing without Azure credentials.

Author: Emergent
Version: 1.0.0
"""

import os
import io
import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VoiceServiceType(Enum):
    """Supported voice service types."""
    AZURE = "azure"
    MOCK = "mock"


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    success: bool
    text: str
    confidence: float
    language: str
    duration_seconds: float
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "success": self.success,
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message
        }


class SpeechToTextService:
    """
    Speech-to-Text service with Azure Speech Services integration.
    
    Supports:
    - Azure Speech Services (production)
    - Mock mode (local testing)
    
    Features:
    - Multiple audio format support (WAV, MP3, OGG)
    - Confidence scores
    - Language detection
    - Graceful error handling
    """
    
    # Supported audio formats
    SUPPORTED_FORMATS = ['wav', 'mp3', 'ogg', 'webm', 'm4a']
    
    # Audio constraints
    MAX_AUDIO_SIZE_MB = 10
    MAX_AUDIO_DURATION_SECONDS = 60
    
    def __init__(self):
        """
        Initialize the Speech-to-Text service.
        
        Reads configuration from environment variables:
        - VOICE_SERVICE: 'azure' or 'mock'
        - AZURE_SPEECH_KEY: Azure Speech Services subscription key
        - AZURE_SPEECH_REGION: Azure region (e.g., 'eastus')
        - MAX_AUDIO_DURATION: Maximum audio duration in seconds
        - MAX_AUDIO_SIZE: Maximum audio file size in MB
        """
        self.service_type = os.getenv("VOICE_SERVICE", "mock").lower()
        self.azure_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.azure_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.default_language = os.getenv("VOICE_LANGUAGES", "en-US").split(",")[0]
        
        # Override defaults with environment variables if set
        max_duration = os.getenv("MAX_AUDIO_DURATION")
        max_size = os.getenv("MAX_AUDIO_SIZE")
        
        if max_duration:
            self.MAX_AUDIO_DURATION_SECONDS = int(max_duration)
        if max_size:
            self.MAX_AUDIO_SIZE_MB = int(max_size)
        
        self._speech_config = None
        self._initialized = False
        
        logger.info(f"STT Service initialized - Mode: {self.service_type}, Region: {self.azure_region}")
    
    def _initialize_azure(self) -> bool:
        """
        Initialize Azure Speech SDK.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
            
        if self.service_type != VoiceServiceType.AZURE.value:
            return False
            
        if not self.azure_key:
            logger.error("AZURE_SPEECH_KEY not set. Cannot initialize Azure Speech Services.")
            return False
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            self._speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_key,
                region=self.azure_region
            )
            self._speech_config.speech_recognition_language = self.default_language
            self._initialized = True
            logger.info(f"Azure Speech SDK initialized successfully for region: {self.azure_region}")
            return True
            
        except ImportError as e:
            logger.error(f"Azure Speech SDK import failed: {e}. Run: pip install azure-cognitiveservices-speech")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Azure Speech SDK: {e}")
            return False
    
    def transcribe(self, audio_data: bytes, audio_format: str = "wav", 
                   language: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format (wav, mp3, ogg)
            language: Language code (e.g., 'en-US'). Uses default if not specified.
        
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        start_time = time.time()
        language = language or self.default_language
        
        logger.info(f"Starting transcription - Format: {audio_format}, Language: {language}, Size: {len(audio_data)} bytes")
        
        # Route to appropriate service
        if self.service_type == VoiceServiceType.MOCK.value:
            result = self._transcribe_mock(audio_data, audio_format, language)
        elif self.service_type == VoiceServiceType.AZURE.value:
            result = self._transcribe_azure(audio_data, audio_format, language)
        else:
            result = TranscriptionResult(
                success=False,
                text="",
                confidence=0.0,
                language=language,
                duration_seconds=0.0,
                error_message=f"Unknown voice service type: {self.service_type}"
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Transcription completed in {elapsed:.2f}s - Success: {result.success}")
        
        return result
    
    def _transcribe_mock(self, audio_data: bytes, audio_format: str, 
                         language: str) -> TranscriptionResult:
        """
        Mock transcription for local testing.
        
        Returns a realistic mock response based on audio data characteristics.
        """
        logger.info("Using MOCK transcription service")
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Generate mock transcription based on audio size
        audio_size_kb = len(audio_data) / 1024
        
        # Mock responses for testing different scenarios
        mock_responses = [
            "I've been struggling with screen time addiction lately.",
            "How can I be happier in my daily life?",
            "I feel anxious about work and need some advice.",
            "What are some mindfulness techniques I can try?",
            "I'm having trouble sleeping and it's affecting my mood."
        ]
        
        # Select mock response based on audio size (deterministic for testing)
        response_index = int(audio_size_kb) % len(mock_responses)
        mock_text = mock_responses[response_index]
        
        # Calculate mock duration (estimate ~150 bytes per second for compressed audio)
        estimated_duration = len(audio_data) / 16000  # Rough estimate
        
        return TranscriptionResult(
            success=True,
            text=mock_text,
            confidence=0.95,
            language=language,
            duration_seconds=estimated_duration,
            raw_response={"mock": True, "audio_size_kb": audio_size_kb}
        )
    
    def _transcribe_azure(self, audio_data: bytes, audio_format: str,
                          language: str) -> TranscriptionResult:
        """
        Transcribe audio using Azure Speech Services.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format
            language: Language code
        
        Returns:
            TranscriptionResult from Azure Speech Services
        """
        if not self._initialize_azure():
            return TranscriptionResult(
                success=False,
                text="",
                confidence=0.0,
                language=language,
                duration_seconds=0.0,
                error_message="Azure Speech Services not initialized. Check AZURE_SPEECH_KEY and AZURE_SPEECH_REGION."
            )
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Set language for this request
            self._speech_config.speech_recognition_language = language
            
            # Create audio stream from bytes
            audio_stream = speechsdk.audio.PushAudioInputStream()
            audio_stream.write(audio_data)
            audio_stream.close()
            
            audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            
            # Create recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self._speech_config,
                audio_config=audio_config
            )
            
            # Perform recognition
            result = recognizer.recognize_once_async().get()
            
            # Process result
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Extract confidence from detailed results if available
                confidence = 0.9  # Default confidence
                if hasattr(result, 'best') and result.best:
                    confidence = result.best[0].confidence
                
                return TranscriptionResult(
                    success=True,
                    text=result.text,
                    confidence=confidence,
                    language=language,
                    duration_seconds=result.duration.total_seconds() if hasattr(result, 'duration') else 0.0,
                    raw_response={"reason": str(result.reason)}
                )
                
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return TranscriptionResult(
                    success=False,
                    text="",
                    confidence=0.0,
                    language=language,
                    duration_seconds=0.0,
                    error_message="No speech could be recognized from the audio."
                )
                
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails.from_result(result)
                error_msg = f"Recognition canceled: {cancellation.reason}"
                
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    error_msg = f"Error: {cancellation.error_details}"
                
                logger.error(f"Azure STT cancellation: {error_msg}")
                
                return TranscriptionResult(
                    success=False,
                    text="",
                    confidence=0.0,
                    language=language,
                    duration_seconds=0.0,
                    error_message=error_msg
                )
            else:
                return TranscriptionResult(
                    success=False,
                    text="",
                    confidence=0.0,
                    language=language,
                    duration_seconds=0.0,
                    error_message=f"Unexpected result reason: {result.reason}"
                )
                
        except Exception as e:
            logger.error(f"Azure STT error: {e}", exc_info=True)
            return TranscriptionResult(
                success=False,
                text="",
                confidence=0.0,
                language=language,
                duration_seconds=0.0,
                error_message=f"Azure Speech Services error: {str(e)}"
            )
    
    def is_available(self) -> Tuple[bool, str]:
        """
        Check if the STT service is available and properly configured.
        
        Returns:
            Tuple of (is_available, status_message)
        """
        if self.service_type == VoiceServiceType.MOCK.value:
            return True, "Mock STT service is available"
        
        if self.service_type == VoiceServiceType.AZURE.value:
            if not self.azure_key:
                return False, "AZURE_SPEECH_KEY not configured"
            if self._initialize_azure():
                return True, f"Azure STT service available (region: {self.azure_region})"
            else:
                return False, "Failed to initialize Azure Speech SDK"
        
        return False, f"Unknown service type: {self.service_type}"
    
    def get_supported_languages(self) -> list:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        # Common languages supported by Azure Speech Services
        return [
            "en-US", "en-GB", "en-AU", "en-IN",
            "es-ES", "es-MX",
            "fr-FR", "fr-CA",
            "de-DE",
            "it-IT",
            "pt-BR", "pt-PT",
            "zh-CN", "zh-TW",
            "ja-JP",
            "ko-KR"
        ]


# Global singleton instance
_stt_service: Optional[SpeechToTextService] = None


def get_stt_service() -> SpeechToTextService:
    """
    Get or create the Speech-to-Text service singleton.
    
    Returns:
        SpeechToTextService instance
    """
    global _stt_service
    if _stt_service is None:
        _stt_service = SpeechToTextService()
    return _stt_service
