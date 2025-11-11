"""
PostgreSQL Connection Handler
Simple username/password authentication for AWE Mental Health Chatbot
"""

from sqlalchemy import create_engine
import os
import logging

logger = logging.getLogger(__name__)


def get_database_engine():
    """
    Create SQLAlchemy engine with simple username/password authentication
    
    Returns:
        SQLAlchemy engine ready for use
        
    Example:
        from backend.database_aad import get_database_engine
        engine = get_database_engine()
    """
    # Get database configuration from environment variables
    host = os.getenv("PGHOST", "awe-database-progres-prod.postgres.database.azure.com")
    database = os.getenv("PGDATABASE", "postgres")
    port = int(os.getenv("PGPORT", "5432"))
    username = os.getenv("PGUSER", "awechatbot")
    password = os.getenv("PGPASSWORD", "AweChatBot@2025!")
    
    logger.info(f"Connecting to PostgreSQL: {host}:{port}/{database} as {username}")
    
    # Simple connection string
    connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}?sslmode=require"
    
    logger.info("Connection string created successfully")
    
    # Create engine with connection pooling
    engine = create_engine(
        connection_string,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,              # Verify connections before using
        pool_recycle=3600,               # Recycle connections every hour
        echo=False,                      # Set to True for SQL debugging
        connect_args={
            "connect_timeout": 10,
            "application_name": "awe-chatbot"
        }
    )
    
    logger.info("✓ PostgreSQL engine created successfully")
    return engine
