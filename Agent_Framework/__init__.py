"""
Gemini Agent Framework - A modular framework for building agents with Google's Gemini Live API
Version: 1.0.0
Current Date and Time (UTC): 2025-04-16 17:36:12
Current User's Login: SAMBITMALLICK2003
"""

# Filter out the specific soundcard warning globally
import warnings
from soundcard.mediafoundation import SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning,
                        message="data discontinuity in recording")

from .agent.base_agent import BaseAgent
from .agent.live_agent import LiveAgent
from .tools.base_tool import BaseTool
from .tools.image_capture import CameraCaptureTool, ScreenCaptureTool
from .tools.calendar_tools import ScheduleMeetingTool, GetMeetingsTool
from .audio.wake_word import WakeWordDetector, SleepWordDetector
from .audio.audio_utils import AudioUtils

__version__ = "1.0.0"