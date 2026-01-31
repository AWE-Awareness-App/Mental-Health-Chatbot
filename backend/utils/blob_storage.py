"""
Azure Blob Storage utility for uploading audio responses.
"""

import os
import uuid
import logging
from typing import Optional
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions

logger = logging.getLogger(__name__)


class AudioBlobStorage:
    """Handles uploading audio files to Azure Blob Storage with SAS tokens."""

    def __init__(self):
        """Initialize blob storage client."""
        self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "voice-responses")

        if not self.account_name or not self.account_key:
            logger.warning("Azure Storage credentials not configured")
            self.client = None
            return

        try:
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            self.client = BlobServiceClient(
                account_url=account_url,
                credential=self.account_key
            )
            self.container_client = self.client.get_container_client(self.container_name)
            logger.info(f"✓ Blob storage initialized: {self.container_name}")
        except Exception as e:
            logger.error(f"Failed to initialize blob storage: {e}")
            self.client = None

    def is_available(self) -> bool:
        """Check if blob storage is configured and available."""
        return self.client is not None

    def upload_audio(
        self,
        audio_data: bytes,
        audio_format: str,
        user_id: Optional[str] = None,
        expiry_hours: int = 24
    ) -> Optional[str]:
        """
        Upload audio file to blob storage and return public URL with SAS token.

        Args:
            audio_data: Audio file bytes
            audio_format: Audio format (mp3, wav, ogg)
            user_id: Optional user identifier for organization
            expiry_hours: Hours until SAS token expires (default 24)

        Returns:
            Public URL with SAS token to the uploaded blob, or None if failed
        """
        if not self.is_available():
            logger.error("Blob storage not available")
            return None

        try:
            # Generate unique blob name
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]

            if user_id:
                # Sanitize user_id for blob naming
                safe_user_id = user_id.replace(":", "_").replace("+", "")[-20:]
                blob_name = f"{safe_user_id}/{timestamp}_{unique_id}.{audio_format}"
            else:
                blob_name = f"responses/{timestamp}_{unique_id}.{audio_format}"

            # Set content type
            content_settings = ContentSettings(
                content_type=self._get_content_type(audio_format),
                cache_control="public, max-age=3600"
            )

            # Upload blob
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                audio_data,
                overwrite=True,
                content_settings=content_settings
            )

            # Generate SAS token for private container
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
            )

            # Build URL with SAS token
            blob_url = f"{blob_client.url}?{sas_token}"

            logger.info(f"✓ Uploaded audio to blob: {blob_name} (SAS expires in {expiry_hours}h)")

            return blob_url

        except Exception as e:
            logger.error(f"Failed to upload audio to blob: {e}", exc_info=True)
            return None

    def _get_content_type(self, audio_format: str) -> str:
        """Get MIME type for audio format."""
        mime_types = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "m4a": "audio/mp4",
            "webm": "audio/webm"
        }
        return mime_types.get(audio_format.lower(), "application/octet-stream")

    def delete_old_blobs(self, days: int = 7) -> int:
        """
        Delete blobs older than specified days (for cleanup).

        Args:
            days: Age threshold in days

        Returns:
            Number of blobs deleted
        """
        if not self.is_available():
            return 0

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            deleted_count = 0

            blobs = self.container_client.list_blobs()
            for blob in blobs:
                if blob.last_modified.replace(tzinfo=None) < cutoff_date:
                    self.container_client.delete_blob(blob.name)
                    deleted_count += 1
                    logger.debug(f"Deleted old blob: {blob.name}")

            if deleted_count > 0:
                logger.info(f"✓ Cleaned up {deleted_count} old audio blobs")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete old blobs: {e}")
            return 0


# Global instance
_blob_storage: Optional[AudioBlobStorage] = None


def get_blob_storage() -> AudioBlobStorage:
    """Get or create global blob storage instance."""
    global _blob_storage
    if _blob_storage is None:
        _blob_storage = AudioBlobStorage()
    return _blob_storage
