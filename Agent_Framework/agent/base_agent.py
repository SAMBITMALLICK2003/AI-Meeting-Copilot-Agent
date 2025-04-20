"""
Base agent class for the Gemini Agent Framework
"""

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents in the framework"""
    
    def __init__(self, name="Agent"):
        self.name = name
        self.tools = {}
        self.conversation_context = []
        
    def add_tool(self, tool):
        """Add a tool to the agent"""
        self.tools[tool.name] = tool
        return self
        
    def get_tool(self, tool_name):
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    @abstractmethod
    async def initialize(self):
        """Initialize the agent"""
        pass
    
    @abstractmethod
    async def run(self):
        """Run the agent"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass