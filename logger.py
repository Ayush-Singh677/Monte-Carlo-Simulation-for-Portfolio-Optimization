import logging
import sys
from config import LOG_FILE, LOG_LEVEL

def setup_logger():
    """Set up the logger for the application."""
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(LOG_FILE, mode='w')

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger() 