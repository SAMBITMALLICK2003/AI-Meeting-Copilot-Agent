"""
Base tool class for the Gemini Agent Framework
"""

from abc import ABC, abstractmethod
from ..utils.logger import setup_logger

class BaseTool(ABC):
    """Base class for all tools that can be used with the Gemini Agent Framework"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.logger = setup_logger(f"gemini_agent_framework.tools.{name}")
        self.logger.debug(f"Initialized {name} tool")
        
    @abstractmethod
    async def execute(self, *args, **kwargs):
        """Execute the tool's functionality"""
        pass
    
    async def initialize(self):
        """Initialize the tool if needed"""
        return True
        
    def __str__(self):
        return f"{self.name}: {self.description}"