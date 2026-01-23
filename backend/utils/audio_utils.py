"""
Audio Utility Functions for AWE Mental Health Chatbot Voice Integration

Provides utilities for audio file validation, format detection,
duration calculation, and encoding/decoding operations.

Author: Emergent
Version: 1.0.0
"""

import os
import io
import base64
import logging
import struct
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    WEBM = "webm"
    M4A = "m4a"
    UNKNOWN = "unknown"


@dataclass
class AudioValidationResult:
    """Result from audio validation."""
    is_valid: bool
    format: str
    size_bytes: int
    size_mb: float
    estimated_duration_seconds: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_mb, 2),
            "estimated_duration_seconds": round(self.estimated_duration_seconds, 2),
            "error_message": self.error_message
        }


# Audio format magic bytes
AUDIO_SIGNATURES = {
    b'RIFF': AudioFormat.WAV,
    b'\xff\xfb': AudioFormat.MP3,
    b'\xff\xfa': AudioFormat.MP3,
    b'\xff\xf3': AudioFormat.MP3,
    b'\xff\xf2': AudioFormat.MP3,
    b'ID3': AudioFormat.MP3,
    b'OggS': AudioFormat.OGG,
    b'\x1aE\xdf\xa3': AudioFormat.WEBM,
    b'ftyp': AudioFormat.M4A,
}


def detect_audio_format(audio_data: bytes) -> AudioFormat:
    """
    Detect audio format from file header bytes.
    
    Args:
        audio_data: Raw audio bytes
    
    Returns:
        Detected AudioFormat enum
    """
    if len(audio_data) < 12:
        return AudioFormat.UNKNOWN
    
    header = audio_data[:12]
    
    # Check for RIFF (WAV)
    if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
        return AudioFormat.WAV
    
    # Check for MP3 (ID3 tag or frame sync)
    if header[:3] == b'ID3':
        return AudioFormat.MP3
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return AudioFormat.MP3
    
    # Check for OGG
    if header[:4] == b'OggS':
        return AudioFormat.OGG
    
    # Check for WebM/Matroska
    if header[:4] == b'\x1aE\xdf\xa3':
        return AudioFormat.WEBM
    
    # Check for M4A/MP4 (ftyp box)
    if header[4:8] == b'ftyp':
        return AudioFormat.M4A
    
    return AudioFormat.UNKNOWN


def estimate_audio_duration(audio_data: bytes, audio_format: AudioFormat) -> float:
    """
    Estimate audio duration from file data.
    
    Args:
        audio_data: Raw audio bytes
        audio_format: Detected audio format
    
    Returns:
        Estimated duration in seconds
    """
    size_bytes = len(audio_data)
    
    if audio_format == AudioFormat.WAV:
        return _estimate_wav_duration(audio_data)
    elif audio_format == AudioFormat.MP3:
        return _estimate_mp3_duration(audio_data)
    elif audio_format == AudioFormat.OGG:
        # Rough estimate: ~128kbps for OGG
        return size_bytes / 16000
    elif audio_format == AudioFormat.WEBM:
        # Rough estimate: ~128kbps
        return size_bytes / 16000
    elif audio_format == AudioFormat.M4A:
        # Rough estimate: ~128kbps
        return size_bytes / 16000
    else:
        # Very rough fallback: assume ~16KB per second
        return size_bytes / 16000


def _estimate_wav_duration(audio_data: bytes) -> float:
    """
    Estimate WAV file duration from header.
    """
    try:
        if len(audio_data) < 44:
            return 0.0
        
        # Parse WAV header
        # Bytes 24-27: Sample rate
        # Bytes 28-31: Byte rate
        sample_rate = struct.unpack('<I', audio_data[24:28])[0]
        byte_rate = struct.unpack('<I', audio_data[28:32])[0]
        
        if byte_rate == 0:
            return 0.0
        
        # Data chunk starts at byte 44 typically
        data_size = len(audio_data) - 44
        duration = data_size / byte_rate
        
        return duration
    except Exception as e:
        logger.warning(f"Error parsing WAV header: {e}")
        return len(audio_data) / 32000  # Fallback estimate


def _estimate_mp3_duration(audio_data: bytes) -> float:
    """
    Estimate MP3 file duration.
    """
    size_bytes = len(audio_data)
    
    # Skip ID3 header if present
    offset = 0
    if audio_data[:3] == b'ID3':
        # ID3v2 header size is in bytes 6-9 (syncsafe integer)
        if len(audio_data) > 10:
            id3_size = (
                (audio_data[6] & 0x7F) << 21 |
                (audio_data[7] & 0x7F) << 14 |
                (audio_data[8] & 0x7F) << 7 |
                (audio_data[9] & 0x7F)
            )
            offset = 10 + id3_size
    
    # Find first frame sync
    while offset < len(audio_data) - 1:
        if audio_data[offset] == 0xFF and (audio_data[offset + 1] & 0xE0) == 0xE0:
            break
        offset += 1
    
    # Rough estimate based on typical bitrates
    # Assume 128kbps average
    data_size = size_bytes - offset
    duration = data_size / 16000  # 128kbps = 16KB/s
    
    return duration


def validate_audio(
    audio_data: bytes,
    max_size_mb: float = 10.0,
    max_duration_seconds: float = 60.0,
    allowed_formats: Optional[list] = None
) -> AudioValidationResult:
    """
    Validate audio data against constraints.
    
    Args:
        audio_data: Raw audio bytes
        max_size_mb: Maximum file size in MB
        max_duration_seconds: Maximum audio duration in seconds
        allowed_formats: List of allowed formats. None = all supported formats.
    
    Returns:
        AudioValidationResult with validation details
    """
    if allowed_formats is None:
        allowed_formats = [f.value for f in AudioFormat if f != AudioFormat.UNKNOWN]
    
    # Check for empty data
    if not audio_data or len(audio_data) == 0:
        return AudioValidationResult(
            is_valid=False,
            format="unknown",
            size_bytes=0,
            size_mb=0.0,
            estimated_duration_seconds=0.0,
            error_message="Audio data is empty"
        )
    
    size_bytes = len(audio_data)
    size_mb = size_bytes / (1024 * 1024)
    
    # Check file size
    if size_mb > max_size_mb:
        return AudioValidationResult(
            is_valid=False,
            format="unknown",
            size_bytes=size_bytes,
            size_mb=size_mb,
            estimated_duration_seconds=0.0,
            error_message=f"Audio file size ({size_mb:.2f}MB) exceeds maximum ({max_size_mb}MB)"
        )
    
    # Detect format
    audio_format = detect_audio_format(audio_data)
    
    if audio_format == AudioFormat.UNKNOWN:
        return AudioValidationResult(
            is_valid=False,
            format="unknown",
            size_bytes=size_bytes,
            size_mb=size_mb,
            estimated_duration_seconds=0.0,
            error_message="Unable to detect audio format. Supported formats: WAV, MP3, OGG, WebM, M4A"
        )
    
    # Check if format is allowed
    if audio_format.value not in allowed_formats:
        return AudioValidationResult(
            is_valid=False,
            format=audio_format.value,
            size_bytes=size_bytes,
            size_mb=size_mb,
            estimated_duration_seconds=0.0,
            error_message=f"Audio format '{audio_format.value}' is not allowed. Allowed formats: {', '.join(allowed_formats)}"
        )
    
    # Estimate duration
    duration = estimate_audio_duration(audio_data, audio_format)
    
    # Check duration
    if duration > max_duration_seconds:
        return AudioValidationResult(
            is_valid=False,
            format=audio_format.value,
            size_bytes=size_bytes,
            size_mb=size_mb,
            estimated_duration_seconds=duration,
            error_message=f"Audio duration ({duration:.1f}s) exceeds maximum ({max_duration_seconds}s)"
        )
    
    # All checks passed
    return AudioValidationResult(
        is_valid=True,
        format=audio_format.value,
        size_bytes=size_bytes,
        size_mb=size_mb,
        estimated_duration_seconds=duration
    )


def audio_to_base64(audio_data: bytes) -> str:
    """
    Encode audio data to base64 string.
    
    Args:
        audio_data: Raw audio bytes
    
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(audio_data).decode('utf-8')


def base64_to_audio(base64_string: str) -> bytes:
    """
    Decode base64 string to audio data.
    
    Args:
        base64_string: Base64 encoded audio
    
    Returns:
        Raw audio bytes
    
    Raises:
        ValueError: If base64 string is invalid
    """
    try:
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 audio data: {e}")


def get_mime_type(audio_format: str) -> str:
    """
    Get MIME type for audio format.
    
    Args:
        audio_format: Audio format string (wav, mp3, ogg, etc.)
    
    Returns:
        MIME type string
    """
    mime_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
        "m4a": "audio/mp4",
    }
    return mime_types.get(audio_format.lower(), "application/octet-stream")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "1:23" or "0:05")
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


class AudioFileHandler:
    """
    Handler for processing audio files from HTTP requests.
    
    Provides utilities for handling file uploads and validation.
    """
    
    def __init__(
        self,
        max_size_mb: float = 10.0,
        max_duration_seconds: float = 60.0,
        allowed_formats: Optional[list] = None
    ):
        """
        Initialize audio file handler.
        
        Args:
            max_size_mb: Maximum file size in MB
            max_duration_seconds: Maximum audio duration in seconds
            allowed_formats: List of allowed audio formats
        """
        self.max_size_mb = max_size_mb
        self.max_duration_seconds = max_duration_seconds
        self.allowed_formats = allowed_formats or ["wav", "mp3", "ogg", "webm", "m4a"]
    
    async def process_upload(self, file) -> Tuple[bytes, AudioValidationResult]:
        """
        Process an uploaded audio file.
        
        Args:
            file: UploadFile from FastAPI
        
        Returns:
            Tuple of (audio_data, validation_result)
        """
        try:
            # Read file content
            audio_data = await file.read()
            
            # Validate
            validation = validate_audio(
                audio_data,
                max_size_mb=self.max_size_mb,
                max_duration_seconds=self.max_duration_seconds,
                allowed_formats=self.allowed_formats
            )
            
            return audio_data, validation
            
        except Exception as e:
            logger.error(f"Error processing audio upload: {e}")
            return b"", AudioValidationResult(
                is_valid=False,
                format="unknown",
                size_bytes=0,
                size_mb=0.0,
                estimated_duration_seconds=0.0,
                error_message=f"Error processing audio file: {str(e)}"
            )
    
    def get_format_from_filename(self, filename: str) -> str:
        """
        Extract audio format from filename.
        
        Args:
            filename: Original filename
        
        Returns:
            Format string or 'unknown'
        """
        if not filename:
            return "unknown"
        
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ""
        
        if ext in self.allowed_formats:
            return ext
        
        return "unknown"
