"""
Azure Entra ID PostgreSQL Connection Handler
Manages token-based authentication for Azure Database for PostgreSQL
Customized for AWE Mental Health Chatbot project structure
"""

from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ClientAuthenticationError
from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL, Engine
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AzureADPostgresConnection:
    """Manages PostgreSQL connection with Azure Entra ID authentication."""
    
    def __init__(self):
        """Initialize Azure AD credential provider."""
        self.credential = DefaultAzureCredential()
        self.scope = "https://ossrdbms-aad.database.windows.net/.default"
        self._last_token: Optional[str] = None
    
    def get_access_token(self) -> str:
        """
        Get a fresh Azure AD access token for PostgreSQL.
        
        Returns:
            str: Valid Azure AD access token
            
        Raises:
            ClientAuthenticationError: If token retrieval fails
        """
        try:
            token = self.credential.get_token(self.scope)
            self._last_token = token.token
            logger.debug("Azure AD token retrieved successfully")
            return token.token
        except ClientAuthenticationError as e:
            logger.error(f"Failed to get Azure AD token: {str(e)}")
            raise Exception(f"Azure AD authentication failed: {str(e)}")
    
    def create_engine(self) -> Engine:
        """
        Create SQLAlchemy engine with Azure AD authentication.
        
        Returns:
            Engine: Configured SQLAlchemy engine with automatic token refresh
        """
        
        # Get database configuration from environment variables
        host = os.getenv("PGHOST", "awe-database-progres-prod.postgres.database.azure.com")
        database = os.getenv("PGDATABASE", "postgres")
        port = int(os.getenv("PGPORT", "5432"))
        username = os.getenv("PGUSER", "awe-chat-bot")
        
        logger.info(f"Connecting to PostgreSQL: {host}:{port}/{database} as {username}")
        
        # Get initial access token
        try:
            password = self.get_access_token()
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            raise
        
        # Build connection URL with sslmode=require for Azure
        password = self.get_access_token()  # ← GET THE TOKEN
        
        # Azure AD requires username@hostname format
        azure_username = f"{username}@{host.split('.')[0]}"
        
        database_url = URL.create(
            "postgresql+psycopg2",
            username=azure_username,
            password=password,
            host=host,
            port=port,
            database=database,
            query={"sslmode": "require"}
        )
        
        # Create engine with connection pooling and token refresh
        # Pool size: 5-10 connections (suitable for single app instance)
        # Pre-ping: ensures stale connections are detected
        # Recycle: PostgreSQL bearer tokens expire ~60 minutes
        engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,              # Verify connections before using
            pool_recycle=3500,               # Recycle connections every ~58 minutes
            echo=False,                      # Set to True for SQL debugging
            connect_args={
                "connect_timeout": 10,
                "application_name": "awe-chatbot"
            }
        )
        
        # Event listener: Refresh token before each new connection
        @event.listens_for(engine, "do_connect")
        def receive_do_connect(dialect, conn_rec, cargs, cparams):
            """
            Refresh token before establishing a new database connection.
            This ensures every connection uses a valid, fresh token.
            """
            try:
                fresh_token = self.get_access_token()
                cparams["password"] = fresh_token
                logger.debug("Fresh token injected for new connection")
            except Exception as e:
                logger.error(f"Failed to refresh token for connection: {str(e)}")
                raise
        
        logger.info("PostgreSQL engine created with Azure AD authentication")
        return engine


def get_database_engine() -> Engine:
    """
    Factory function to create and return a configured database engine.
    
    Returns:
        Engine: SQLAlchemy engine ready for use
        
    Example:
        from backend.database_aad import get_database_engine
        engine = get_database_engine()
    """
    aad_conn = AzureADPostgresConnection()
    return aad_conn.create_engine()
