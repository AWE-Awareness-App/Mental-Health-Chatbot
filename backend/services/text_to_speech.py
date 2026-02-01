"""
Text-to-Speech Service for AWE Mental Health Chatbot

Provides Azure Speech Services integration for synthesizing text to audio.
Supports multiple voices and audio formats.
Includes mock mode for local testing without Azure credentials.

Author: Emergent
Version: 1.0.0
"""

import os
import io
import logging
import time
import base64
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio output formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"


@dataclass
class VoiceInfo:
    """Information about an available voice."""
    name: str
    display_name: str
    language: str
    gender: str
    neural: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "language": self.language,
            "gender": self.gender,
            "neural": self.neural
        }


@dataclass
class SynthesisResult:
    """Result from text-to-speech synthesis."""
    success: bool
    audio_data: Optional[bytes]
    audio_format: str
    duration_seconds: float
    voice_used: str
    error_message: Optional[str] = None
    audio_base64: Optional[str] = None
    
    def to_dict(self, include_audio: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "success": self.success,
            "audio_format": self.audio_format,
            "duration_seconds": self.duration_seconds,
            "voice_used": self.voice_used,
            "error_message": self.error_message
        }
        if include_audio and self.audio_base64:
            result["audio_base64"] = self.audio_base64
        return result


class TextToSpeechService:
    """
    Text-to-Speech service with Azure Speech Services integration.
    
    Supports:
    - Azure Speech Services (production)
    - Mock mode (local testing)
    
    Features:
    - Multiple voice options (neural voices)
    - Multiple output formats (MP3, WAV, OGG)
    - Text length validation
    - Graceful error handling
    """
    
    # Text constraints
    MAX_TEXT_LENGTH = 5000  # Characters
    MIN_TEXT_LENGTH = 1
    
    # Default voice
    DEFAULT_VOICE = "en-US-AriaNeural"
    
    # Available voices (Azure Neural voices)
    AVAILABLE_VOICES = [
        # Indian English voices
        VoiceInfo("en-IN-NeerjaNeural", "Neerja (India Female)", "en-IN", "Female"),
        VoiceInfo("en-IN-PrabhatNeural", "Prabhat (India Male)", "en-IN", "Male"),

        # US English voices
        VoiceInfo("en-US-AriaNeural", "Aria (US Female)", "en-US", "Female"),
        VoiceInfo("en-US-GuyNeural", "Guy (US Male)", "en-US", "Male"),
        VoiceInfo("en-US-JennyNeural", "Jenny (US Female)", "en-US", "Female"),
        VoiceInfo("en-US-DavisNeural", "Davis (US Male)", "en-US", "Male"),

        # UK English voices
        VoiceInfo("en-GB-SoniaNeural", "Sonia (UK Female)", "en-GB", "Female"),
        VoiceInfo("en-GB-RyanNeural", "Ryan (UK Male)", "en-GB", "Male"),

        # Australian English voices
        VoiceInfo("en-AU-NatashaNeural", "Natasha (AU Female)", "en-AU", "Female"),

        # Spanish voices
        VoiceInfo("es-ES-ElviraNeural", "Elvira (ES Female)", "es-ES", "Female"),
        VoiceInfo("es-MX-DaliaNeural", "Dalia (MX Female)", "es-MX", "Female"),

        # French voices
        VoiceInfo("fr-FR-DeniseNeural", "Denise (FR Female)", "fr-FR", "Female"),
    ]
    
    def __init__(self):
        """
        Initialize the Text-to-Speech service.
        
        Reads configuration from environment variables:
        - VOICE_SERVICE: 'azure' or 'mock'
        - AZURE_SPEECH_KEY: Azure Speech Services subscription key
        - AZURE_SPEECH_REGION: Azure region (e.g., 'eastus')
        - DEFAULT_VOICE: Default voice name
        - AUDIO_FORMAT: Default audio format (mp3, wav, ogg)
        """
        self.service_type = os.getenv("VOICE_SERVICE", "mock").lower()
        self.azure_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.azure_region = os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.default_voice = os.getenv("DEFAULT_VOICE", self.DEFAULT_VOICE)
        self.default_format = os.getenv("AUDIO_FORMAT", "mp3").lower()
        
        self._speech_config = None
        self._initialized = False
        
        logger.info(f"TTS Service initialized - Mode: {self.service_type}, Voice: {self.default_voice}")
    
    def _preprocess_text_for_speech(self, text: str) -> str:
        """
        Preprocess text to improve speech synthesis pronunciation.

        Fixes common pronunciation issues:
        - "7Cs" -> "7 Cs" (pronounced as "seven Cs")
        - "8Ps" -> "8 Ps" (pronounced as "eight Ps")

        Note: Leaves "4 Aces" and other spaced patterns unchanged.

        Args:
            text: Original text

        Returns:
            Preprocessed text with improved pronunciation hints
        """
        import re

        # Fix specific number + letter + 's' patterns without spaces
        # Only matches patterns like "7Cs", "8Ps" where there's NO space
        # This won't affect "4 Aces" which already has a space
        text = re.sub(r'(\d+)([A-Z])s\b', r'\1 \2s', text)

        return text

    def _initialize_azure(self) -> bool:
        """
        Initialize Azure Speech SDK.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
            
        if self.service_type != "azure":
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
            
            # Set default output format to MP3
            self._speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
            )
            
            self._initialized = True
            logger.info(f"Azure Speech SDK initialized successfully for TTS (region: {self.azure_region})")
            return True
            
        except ImportError as e:
            logger.error(f"Azure Speech SDK import failed: {e}. Run: pip install azure-cognitiveservices-speech")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Azure Speech SDK: {e}")
            return False
    
    def synthesize(self, text: str, voice: Optional[str] = None,
                   audio_format: str = "mp3") -> SynthesisResult:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize
            voice: Voice name (e.g., 'en-US-AriaNeural'). Uses default if not specified.
            audio_format: Output format (mp3, wav, ogg)

        Returns:
            SynthesisResult with audio data and metadata
        """
        start_time = time.time()
        voice = voice or self.default_voice
        audio_format = audio_format.lower()

        logger.info(f"Starting synthesis - Voice: {voice}, Format: {audio_format}, Text length: {len(text)}")

        # Preprocess text for better pronunciation
        text = self._preprocess_text_for_speech(text)

        # Validate text
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return SynthesisResult(
                success=False,
                audio_data=None,
                audio_format=audio_format,
                duration_seconds=0.0,
                voice_used=voice,
                error_message="Text is too short or empty"
            )
        
        if len(text) > self.MAX_TEXT_LENGTH:
            return SynthesisResult(
                success=False,
                audio_data=None,
                audio_format=audio_format,
                duration_seconds=0.0,
                voice_used=voice,
                error_message=f"Text exceeds maximum length of {self.MAX_TEXT_LENGTH} characters"
            )
        
        # Route to appropriate service
        if self.service_type == "mock":
            result = self._synthesize_mock(text, voice, audio_format)
        elif self.service_type == "azure":
            result = self._synthesize_azure(text, voice, audio_format)
        else:
            result = SynthesisResult(
                success=False,
                audio_data=None,
                audio_format=audio_format,
                duration_seconds=0.0,
                voice_used=voice,
                error_message=f"Unknown voice service type: {self.service_type}"
            )
        
        elapsed = time.time() - start_time
        logger.info(f"Synthesis completed in {elapsed:.2f}s - Success: {result.success}")
        
        return result
    
    def _synthesize_mock(self, text: str, voice: str, 
                         audio_format: str) -> SynthesisResult:
        """
        Mock synthesis for local testing.
        
        Generates a minimal valid audio file for testing purposes.
        """
        logger.info("Using MOCK TTS service")
        
        # Simulate processing time (roughly 0.1s per 50 characters)
        processing_time = max(0.2, len(text) / 500)
        time.sleep(min(processing_time, 1.0))  # Cap at 1 second
        
        # Generate mock audio data
        # This is a minimal valid MP3 frame header for testing
        # In real usage, Azure would provide proper audio
        mock_audio = self._generate_mock_audio(audio_format)
        
        # Estimate duration (roughly 150 words per minute, 5 characters per word)
        estimated_words = len(text) / 5
        estimated_duration = estimated_words / 150 * 60  # seconds
        
        # Encode to base64
        audio_base64 = base64.b64encode(mock_audio).decode('utf-8')
        
        return SynthesisResult(
            success=True,
            audio_data=mock_audio,
            audio_format=audio_format,
            duration_seconds=estimated_duration,
            voice_used=voice,
            audio_base64=audio_base64
        )
    
    def _generate_mock_audio(self, audio_format: str) -> bytes:
        """
        Generate minimal mock audio data for testing.
        
        Returns bytes representing a minimal valid audio file.
        """
        if audio_format == "mp3":
            # Minimal MP3 frame (silent frame)
            # This is a valid MP3 frame that produces silence
            return bytes([
                0xFF, 0xFB, 0x90, 0x00,  # MP3 header
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
            ] * 100)  # Repeat for ~1 second of audio
            
        elif audio_format == "wav":
            # Minimal WAV header with silence
            import struct
            sample_rate = 16000
            num_samples = sample_rate  # 1 second
            
            # WAV header
            header = b'RIFF'
            header += struct.pack('<I', 36 + num_samples * 2)  # File size
            header += b'WAVE'
            header += b'fmt '
            header += struct.pack('<I', 16)  # Chunk size
            header += struct.pack('<H', 1)   # PCM format
            header += struct.pack('<H', 1)   # Mono
            header += struct.pack('<I', sample_rate)  # Sample rate
            header += struct.pack('<I', sample_rate * 2)  # Byte rate
            header += struct.pack('<H', 2)   # Block align
            header += struct.pack('<H', 16)  # Bits per sample
            header += b'data'
            header += struct.pack('<I', num_samples * 2)  # Data size
            
            # Add silence
            audio_data = header + bytes(num_samples * 2)
            return audio_data
            
        else:  # ogg or other
            # Return minimal placeholder
            return b'OggS' + bytes(100)
    
    def _synthesize_azure(self, text: str, voice: str,
                          audio_format: str) -> SynthesisResult:
        """
        Synthesize text using Azure Speech Services.
        
        Args:
            text: Text to synthesize
            voice: Voice name
            audio_format: Output format
        
        Returns:
            SynthesisResult from Azure Speech Services
        """
        if not self._initialize_azure():
            return SynthesisResult(
                success=False,
                audio_data=None,
                audio_format=audio_format,
                duration_seconds=0.0,
                voice_used=voice,
                error_message="Azure Speech Services not initialized. Check AZURE_SPEECH_KEY and AZURE_SPEECH_REGION."
            )
        
        try:
            import azure.cognitiveservices.speech as speechsdk
            
            # Set voice
            self._speech_config.speech_synthesis_voice_name = voice
            
            # Set output format based on request
            if audio_format == "mp3":
                self._speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
                )
            elif audio_format == "wav":
                self._speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
                )
            else:
                self._speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
                )
                audio_format = "mp3"  # Default to MP3
            
            # Create synthesizer with audio output to memory
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config,
                audio_config=None  # Output to stream instead of speaker
            )
            
            # Perform synthesis
            result = synthesizer.speak_text_async(text).get()
            
            # Process result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # Estimate duration from audio data size
                # For 32kbps MP3: bytes / (32000/8) = seconds
                if audio_format == "mp3":
                    duration = len(audio_data) / 4000  # 32kbps = 4000 bytes/sec
                else:
                    duration = len(audio_data) / 32000  # 16kHz 16-bit = 32000 bytes/sec
                
                return SynthesisResult(
                    success=True,
                    audio_data=audio_data,
                    audio_format=audio_format,
                    duration_seconds=duration,
                    voice_used=voice,
                    audio_base64=audio_base64
                )
                
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
                error_msg = f"Synthesis canceled: {cancellation.reason}"
                
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    error_msg = f"Error: {cancellation.error_details}"
                
                logger.error(f"Azure TTS cancellation: {error_msg}")
                
                return SynthesisResult(
                    success=False,
                    audio_data=None,
                    audio_format=audio_format,
                    duration_seconds=0.0,
                    voice_used=voice,
                    error_message=error_msg
                )
            else:
                return SynthesisResult(
                    success=False,
                    audio_data=None,
                    audio_format=audio_format,
                    duration_seconds=0.0,
                    voice_used=voice,
                    error_message=f"Unexpected result reason: {result.reason}"
                )
                
        except Exception as e:
            logger.error(f"Azure TTS error: {e}", exc_info=True)
            return SynthesisResult(
                success=False,
                audio_data=None,
                audio_format=audio_format,
                duration_seconds=0.0,
                voice_used=voice,
                error_message=f"Azure Speech Services error: {str(e)}"
            )
    
    def get_available_voices(self, language_filter: Optional[str] = None) -> List[VoiceInfo]:
        """
        Get list of available voices.
        
        Args:
            language_filter: Optional language code to filter voices (e.g., 'en-US')
        
        Returns:
            List of VoiceInfo objects
        """
        voices = self.AVAILABLE_VOICES
        
        if language_filter:
            voices = [v for v in voices if v.language.startswith(language_filter.split('-')[0])]
        
        return voices
    
    def is_available(self) -> Tuple[bool, str]:
        """
        Check if the TTS service is available and properly configured.
        
        Returns:
            Tuple of (is_available, status_message)
        """
        if self.service_type == "mock":
            return True, "Mock TTS service is available"
        
        if self.service_type == "azure":
            if not self.azure_key:
                return False, "AZURE_SPEECH_KEY not configured"
            if self._initialize_azure():
                return True, f"Azure TTS service available (region: {self.azure_region})"
            else:
                return False, "Failed to initialize Azure Speech SDK"
        
        return False, f"Unknown service type: {self.service_type}"


# Global singleton instance
_tts_service: Optional[TextToSpeechService] = None


def get_tts_service() -> TextToSpeechService:
    """
    Get or create the Text-to-Speech service singleton.
    
    Returns:
        TextToSpeechService instance
    """
    global _tts_service
    if _tts_service is None:
        _tts_service = TextToSpeechService()
    return _tts_service
