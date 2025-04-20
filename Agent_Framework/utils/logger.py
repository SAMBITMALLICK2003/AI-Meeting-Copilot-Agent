"""
Centralized logging configuration for the Gemini Agent Framework
"""

import logging
import os
import sys
from datetime import datetime

# Default log levels for different components
DEFAULT_LOG_LEVEL = logging.INFO
TOOL_LOG_LEVEL = logging.INFO
AUDIO_LOG_LEVEL = logging.INFO
AGENT_LOG_LEVEL = logging.INFO

# Log format with timestamp, level, and component
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Current timestamp for log file naming
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_logger(name):
    """
    Set up a logger with the given name
    
    Args:
        name: Name of the logger (typically module name)
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # Determine log level based on component
    if name.startswith('gemini_agent_framework.tools'):
        logger.setLevel(TOOL_LOG_LEVEL)
    elif name.startswith('gemini_agent_framework.audio'):
        logger.setLevel(AUDIO_LOG_LEVEL)
    elif name.startswith('gemini_agent_framework.agent'):
        logger.setLevel(AGENT_LOG_LEVEL)
    else:
        logger.setLevel(DEFAULT_LOG_LEVEL)
    
    # Only add handlers if they don't exist already
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(f'logs/gemini_{timestamp}.log')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    
    return logger