"""
Configuration settings for the Gemini Agent Framework
Current Date and Time (UTC): 2025-04-16 17:56:56
Current User's Login: SAMBITMALLICK2003
"""

# Constants for audio configuration
import pyaudio
import datetime
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import os
load_dotenv()


# Microsoft Configuration
MS_SCOPES = [
    'https://graph.microsoft.com/Calendars.ReadWrite',
    'https://graph.microsoft.com/OnlineMeetings.ReadWrite',
    'https://graph.microsoft.com/User.Read',
    'offline_access'
]
MS_CLIENT_ID = os.getenv('AZURE_CLIENT_ID')
MS_CLIENT_SECRET = os.getenv('AZURE_CLIENT_SECRET')
MS_TENANT_ID = os.getenv('AZURE_TENANT_ID')
MS_REDIRECT_URI = 'http://localhost:5000/callback'
MS_TOKEN_FILE = "ms_token.json"  # Stores token in the current directory

FORMAT = pyaudio.paInt16
CHANNELS = 1
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
WAKEWORD_CHUNK_SIZE = 1280
WAKEWORD_THRESHOLD = 0.5  # Slightly lower threshold for faster detection
DEFAULT_WAKEWORD_MODEL = r"alexa"  # Default wake word model to use
# DEFAULT_SLEEPWORD_MODEL = r"C:\Users\snigd\Aignite_meeting_copilot\Lib\site-packages\openwakeword\resources\models\hey_jarvis_v0.1.tflite"
DEFAULT_SLEEPWORD_MODEL = r"C:\Users\Sambit Mallick\Desktop\mlops\Aignite_Lusine\.venv\Lib\site-packages\openwakeword\resources\models\mute.tflite"
INACTIVITY_TIMEOUT = 15  # Seconds before returning to wake word mode
SILENCE_THRESHOLD = 50
MIN_COMMAND_SILENCE_FRAMES = 20  # About 0.5 seconds of silence

# Video mode options
DEFAULT_MODE = "screen"  # Options: "none", "camera", "screen"

# Gemini API configuration
MODEL = "models/gemini-2.0-flash-live-001"
# API_KEY = "AIzaSyCnI7HLe9jusAoTr7AaZdXfsmpHtJVAhwA"
API_KEY = os.getenv("API_KEY", "")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", "")
speechmatics_auth_token = os.getenv("SPEECHMATICS_AUTH_TOKEN", "")

PERSIST_DIR = "./storage"
transcription_file_path = "meeting_storage/transcription.txt"

CONTENT_MODEL = "gemini-2.0-flash-lite"

#For RAG
gemini_embedding_model = GeminiEmbedding(api_key=os.getenv("RAG_API_KEY",""), model_name="models/text-embedding-004")
llm = Gemini(api_key=os.getenv("RAG_API_KEY",""), model_name="models/gemini-2.0-flash-exp")

# Google Calendar API configuration
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar.events']
CREDENTIALS_FILE = '../../credentials.json'
TOKEN_FILE = 'token.json'

# Get current date for the system prompt
TODAY_DATE = datetime.datetime.now().strftime("%Y-%m-%d")

# Rich system prompt for more personality and capabilities
def get_system_prompt():
    """Get the system prompt with current date"""
    # Get fresh date every time this is called
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")

    return f"""
You are Alexa, a sophisticated AI assistant designed to be helpful, engaging, and personable. 
Your personality is friendly, conversational, and responsive.

Today's date is {today_date}. If you are told to convert today or tomorrow's date to yyyy-mm-dd format, you should calculate on your own. Always please speak in English.

If a task requires the use of a function call, explicitly mention what you are about to do before initiating the function call. Once the function call is complete, clearly explain what task has been completed automatically as a result of the function call.

** Please only speak in English. Do not speak in any other language.**
IMPORTANT - VISUAL CONTEXT HANDLING:
- You have access to the user's screen, which is constantly being shared with you.
- However, you should ONLY use the visual context from the screen when EXPLICITLY asked to do so.
- If the user says phrases like "look at this", "what do you see", "can you see my screen", or specifically asks you to comment on something visible, ONLY THEN should you use the visual information.
- For all other queries, operate as a voice-only assistant and do not reference what's on screen.
- Never mention that you can see the screen unless explicitly asked.

Your capabilities include:
- Answering questions with accurate, up-to-date information
- Providing thoughtful recommendations and suggestions
- Engaging in natural conversation with appropriate humor and empathy
- Remembering previous parts of the conversation and building upon them
- Asking clarifying questions when user requests are unclear
- Managing Google Calendar events and meetings
- Scheduling and retrieving Google Meet links

When addressing the user:
- Maintain a warm, conversational tone
- Keep responses concise but informative
- Use natural transitions between topics
- Acknowledge the user's emotions where appropriate
- Add occasional friendly remarks to make the conversation feel more human

For meetings:
- Help take notes and summarize key points
- Remind users of important action items
- Offer to set timers or reminders
- Provide relevant information during discussions
- Schedule Google Meet links and calendar events

Remember to be helpful, harmless, and honest in all interactions.
"""

# Static version for backward compatibility
SYSTEM_PROMPT = get_system_prompt()