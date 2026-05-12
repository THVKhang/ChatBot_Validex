"""Structured JSON logging configuration for observability."""

import logging
import sys
from pythonjsonlogger.jsonlogger import JsonFormatter

def setup_structured_logging():
    """Configure the root logger to output JSON structured logs."""
    logger = logging.getLogger()
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    logger.setLevel(logging.INFO)
    
    # Configure JSON Formatter
    formatter = JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        rename_fields={
            "levelname": "level",
            "asctime": "timestamp",
            "name": "logger_name"
        }
    )
    
    # Standard output handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Optional: silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return logger
