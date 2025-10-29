"""
Startup validation utilities.
Validates environment variables and system requirements before starting the chatbot.
"""

import os
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Required environment variables
REQUIRED_ENV_VARS = [
    "DATABASE_URL",
    "OPENAI_API_KEY",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_WHATSAPP_NUMBER"
]

# Placeholder values that indicate env var is not configured
PLACEHOLDER_VALUES = [
    "your_openai_api_key_here",
    "your_twilio_account_sid_here",
    "your_twilio_auth_token_here",
    "sk-your-openai-api-key",
    "your_actual_sid",
    "your_actual_token",
    "your_actual_key"
]


def validate_environment() -> Tuple[bool, List[str]]:
    """
    Validate all required environment variables are set.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    for var_name in REQUIRED_ENV_VARS:
        value = os.getenv(var_name)
        
        # Check if variable exists
        if not value:
            errors.append(f"❌ {var_name} is not set")
            continue
        
        # Check if it's a placeholder value
        if value.lower() in [p.lower() for p in PLACEHOLDER_VALUES]:
            errors.append(f"⚠️ {var_name} appears to be a placeholder value")
            continue
        
        # Specific validations
        if var_name == "OPENAI_API_KEY":
            if not (value.startswith("sk-") or value.startswith("sk-proj-")):
                errors.append(f"⚠️ {var_name} doesn't look like a valid OpenAI key")
        
        if var_name == "DATABASE_URL":
            if not value.startswith("postgresql://"):
                errors.append(f"⚠️ {var_name} doesn't look like a valid PostgreSQL URL")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_openai_key(api_key: str) -> bool:
    """
    Validate OpenAI API key by making a test request.
    
    Returns:
        True if key is valid, False otherwise
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Try to list models as a simple validation
        client.models.list()
        logger.info("✅ OpenAI API key validated successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ OpenAI API key validation failed: {e}")
        return False


def validate_database_connection() -> bool:
    """
    Validate database connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        from database import engine
        from sqlalchemy import text
        
        # Try to connect
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
        
        logger.info("✅ Database connection validated successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def check_pgvector_extension() -> bool:
    """
    Check if pgvector extension is installed in PostgreSQL.
    
    Returns:
        True if extension exists, False otherwise
    """
    try:
        from database import engine
        from sqlalchemy import text
        
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            has_extension = result.fetchone()[0]
        
        if has_extension:
            logger.info("✅ pgvector extension is installed")
        else:
            logger.error("❌ pgvector extension is NOT installed")
        
        return has_extension
    
    except Exception as e:
        logger.error(f"❌ Error checking pgvector extension: {e}")
        return False


def run_all_startup_checks() -> Dict:
    """
    Run all startup validation checks.
    
    Returns:
        Dictionary with check results
    """
    logger.info("🔍 Running startup validation checks...")
    
    results = {
        "environment": {"valid": False, "errors": []},
        "database": {"connected": False},
        "pgvector": {"installed": False},
        "openai": {"valid": False}
    }
    
    # Check environment variables
    env_valid, env_errors = validate_environment()
    results["environment"]["valid"] = env_valid
    results["environment"]["errors"] = env_errors
    
    if not env_valid:
        logger.warning("⚠️ Environment validation found issues:")
        for error in env_errors:
            logger.warning(f"  {error}")
    else:
        logger.info("✅ All environment variables validated")
    
    # Check database connection
    results["database"]["connected"] = validate_database_connection()
    
    # Check pgvector extension
    if results["database"]["connected"]:
        results["pgvector"]["installed"] = check_pgvector_extension()
    
    # Validate OpenAI key (optional - skip if placeholder)
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key not in PLACEHOLDER_VALUES:
        results["openai"]["valid"] = validate_openai_key(openai_key)
    else:
        logger.warning("⚠️ Skipping OpenAI API key validation (not configured)")
    
    logger.info("✅ Startup validation checks complete")
    return results
