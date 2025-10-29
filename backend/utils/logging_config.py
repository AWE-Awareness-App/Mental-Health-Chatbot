"""
Enhanced logging configuration for the therapeutic chatbot.
Provides structured logging with timestamps and detailed formatting.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Get log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create file handler (main log)
    log_file = log_path / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Create error file handler
    error_log_file = log_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)
    
    # Set level for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logging.info(f"✅ Logging configured: level={log_level}, log_dir={log_dir}")


def log_startup_banner():
    """Log a startup banner with system information."""
    logger = logging.getLogger(__name__)
    
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       WhatsApp Therapeutic Chatbot - Production MVP         ║
║              Digital Wellness & Mental Health                ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    logger.info(banner)
    logger.info("🚀 Starting chatbot initialization...")
