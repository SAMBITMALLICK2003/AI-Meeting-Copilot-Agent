"""
LiveAgent implementation for the Gemini Agent Framework
"""

import asyncio
import time
import argparse
import datetime
import logging
from google import genai
from google.genai import types
from ..agent.base_agent import BaseAgent
from ..audio.wake_word import WakeWordDetector, SleepWordDetector
from ..audio.voice_capture import VoiceCapture
from ..tools.image_capture import CameraCaptureTool, ScreenCaptureTool
from ..tools.calendar_tools import ScheduleMeetingTool, GetMeetingsTool
from ..tools.rag_tools import GetQueryTool
from ..tools.meeting_content_tools import GetContentTool
from ..config.settings import (
    MODEL,
    API_KEY,
    get_system_prompt,
    DEFAULT_WAKEWORD_MODEL,
    DEFAULT_SLEEPWORD_MODEL,
    DEFAULT_MODE,
    INACTIVITY_TIMEOUT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LiveAgent(BaseAgent):
    """Agent implementation using Gemini Live API"""

    def __init__(self,
                name="Gemini Assistant",
                api_key=API_KEY,
                model_name=MODEL,
                system_prompt=None,
                video_mode=DEFAULT_MODE,
                wakeword_model=DEFAULT_WAKEWORD_MODEL,
                sleepword_model=DEFAULT_SLEEPWORD_MODEL,
                inactivity_timeout=INACTIVITY_TIMEOUT,
                enable_calendar=True):
        """
        Initialize the LiveAgent

        Args:
            name: Name of the agent
            api_key: Gemini API key
            model_name: Name of the Gemini model to use
            system_prompt: System prompt for the agent (if None, gets current date)
            video_mode: Video mode to use (none, camera, screen)
            wakeword_model: Wake word model to use
            sleepword_model: Sleep word model to use
            inactivity_timeout: Seconds of inactivity before returning to wake word mode
            enable_calendar: Whether to enable Google Calendar tools
        """
        super().__init__(name)
        self.api_key = api_key
        self.model_name = model_name
        # If system_prompt is None, get the current system prompt with today's date
        self.system_prompt = system_prompt if system_prompt is not None else get_system_prompt()
        self.video_mode = video_mode
        self.wakeword_model = wakeword_model
        self.sleepword_model = sleepword_model
        self.inactivity_timeout = inactivity_timeout
        self.enable_calendar = enable_calendar

        # State variables
        self.wakeword_active = True
        self.speaking = False
        self.last_activity_time = time.time()

        # Components
        self.wakeword_detector = None
        self.sleepword_detector = None
        self.voice_capture = None

        # Gemini API
        self.client = None
        self.session = None

        # Queues for communication
        self.audio_in_queue = None
        self.out_queue = None

        # Today's date for tool functions
        self.today_date = datetime.datetime.now().strftime('%Y-%m-%d')

    def _create_tools_config(self):
        """Create the tools configuration for Gemini API"""
        today_date = self.today_date

        # Define the tool declarations for Gemini API
        tool_declarations = [
            # Calendar scheduling tool
            types.FunctionDeclaration(
                name="schedule_google_meet",
                description="Schedules a meeting in the user's primary Google Calendar and generates a Google Meet link. Requires date and time.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "date_str": types.Schema(
                            type=types.Type.STRING,
                            description=f"REQUIRED. Today's date is {today_date}. The specific date for the meeting. You MUST resolve relative dates like 'today', 'tomorrow', 'next Friday' based on the current date provided (e.g., {today_date}) and provide the result EXCLUSIVELY in 'YYYY-MM-DD' format."
                        ),
                        "time_str": types.Schema(
                            type=types.Type.STRING,
                            description="Time of the meeting in HH:MM:SS format (24-hour clock). Gemini should parse natural language times (e.g., '3 PM', '14:00')."
                        ),
                        "summary": types.Schema(
                            type=types.Type.STRING,
                            description="Required title or summary for the meeting."
                        ),
                        "description": types.Schema(
                            type=types.Type.STRING,
                            description="Optional description or agenda for the meeting."
                        ),
                        "duration_minutes": types.Schema(
                            type=types.Type.INTEGER,
                            description="Optional duration of the meeting in minutes. Gemini can parse phrases like 'an hour' (60) or '30 minutes'. Defaults to 60 minutes if not specified."
                        ),
                        "attendees": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING),
                            description="Optional list of email addresses for attendees to invite. Often omitted in user prompts."
                        ),
                        "timezone": types.Schema(
                            type=types.Type.STRING,
                            description="Optional timezone for the meeting time. Gemini should parse if specified (e.g., '3 PM Eastern'). Defaults to 'IST' if not specified or not parsable."
                        ),
                    },
                    required=["date_str", "time_str", "summary"]
                )
            ),
            # Get meetings tool
            types.FunctionDeclaration(
                name="get_scheduled_meetings",
                description="Retrieves scheduled events (meetings) from the user's Google Calendar within a specified date and time range.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "start_date_str": types.Schema(
                            type=types.Type.STRING,
                            description=f"REQUIRED. The start date for the search range. Today is {today_date}. Resolve relative dates (e.g., 'today', 'yesterday', 'next Monday') to 'YYYY-MM-DD' format."
                        ),
                        "start_time_str": types.Schema(
                            type=types.Type.STRING,
                            description="REQUIRED. The start time for the search range in 'HH:MM:SS' format (24-hour). Use '00:00:00' for the beginning of the day if only a date is specified."
                        ),
                        "end_date_str": types.Schema(
                            type=types.Type.STRING,
                            description=f"REQUIRED. The end date for the search range. Today is {today_date}. Resolve relative dates to 'YYYY-MM-DD' format."
                        ),
                        "end_time_str": types.Schema(
                            type=types.Type.STRING,
                            description="REQUIRED. The end time for the search range in 'HH:MM:SS' format (24-hour). Use '23:59:59' for the end of the day if only a date is specified."
                        ),
                        "timezone": types.Schema(
                            type=types.Type.STRING,
                            description="Optional timezone for interpreting the start/end times (e.g., 'America/Los_Angeles', 'Asia/Kolkata', 'UTC'). Must be a valid IANA Time Zone Database name. Defaults to 'IST' if not specified."
                        ),
                        "calendar_id": types.Schema(
                            type=types.Type.STRING,
                            description="Optional calendar ID to search. Defaults to 'primary' (the user's main calendar)."
                        ),
                        "max_results": types.Schema(
                            type=types.Type.INTEGER,
                            description="Optional maximum number of events to return. Defaults to 50."
                        ),
                    },
                    required=["start_date_str", "start_time_str", "end_date_str", "end_time_str"]
                )
            ),
            # Additional tool declarations can be added here
            types.FunctionDeclaration(
                name="query_docs",
                description="Searches locally indexed documents (like previously uploaded PDFs or text files) to answer questions based only on their content. Use this tool when the user asksto find information 'in the document', 'from the uploaded files', 'in my notes', 'what the record says about X', 'based on the provided context', or refers to 'past records' or the 'knowledge base' associated with this session. This searches the specific documents that have been processed and stored locally. Do NOT use for general web searches, code execution, or real-time information not present in the indexed files.",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "query": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description="The natural language query string used to search the document index."
                        ),
                    },
                ),
            ),
            types.FunctionDeclaration(
                name="generate_meeting_content",
                description="Generates different types of meeting content such as summary, minutes, or action items based on the provided query.",
                parameters=genai.types.Schema(
                    type=genai.types.Type.OBJECT,
                    properties={
                        "query": genai.types.Schema(
                            type=genai.types.Type.STRING,
                            description=(
                                "A natural language query indicating what to generate from the meeting transcript. "
                                "Examples: 'generate summary', 'generate minutes of the meeting', 'generate action items', etc."
                            ),
                        ),
                    },
                ),
            )
        ]

        # Create the tools configuration
        tools = [
            types.Tool(code_execution=types.ToolCodeExecution),
            types.Tool(google_search=types.GoogleSearch()),
            types.Tool(
                function_declarations=tool_declarations
            )
        ]

        return tools

    async def initialize(self):
        """Initialize the agent"""
        logger.info(f"Initializing {self.name}...")

        # Initialize queues
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)

        # Initialize voice capture
        self.voice_capture = VoiceCapture()

        # Initialize wake word detector
        logger.info("Initializing wake word detector...")
        self.wakeword_detector = WakeWordDetector(self.wakeword_model)
        self.wakeword_detector.start()

        # Initialize sleep word detector
        logger.info("Initializing sleep word detector...")
        self.sleepword_detector = SleepWordDetector(self.sleepword_model)
        self.sleepword_detector.start()

        # Initialize video capture tools based on mode
        if self.video_mode == "camera":
            logger.info("Camera mode enabled. Initializing camera...")
            camera_tool = CameraCaptureTool()
            await camera_tool.initialize()
            self.add_tool(camera_tool)
        elif self.video_mode == "screen":
            logger.info("Screen capture mode enabled. Initializing screen capture...")
            self.add_tool(ScreenCaptureTool())

        # Initialize Google Calendar tools if enabled
        if self.enable_calendar:
            logger.info("Google Calendar integration enabled. Initializing calendar tools...")

            # Add schedule meeting tool
            schedule_meeting_tool = ScheduleMeetingTool()
            await schedule_meeting_tool.initialize()
            self.add_tool(schedule_meeting_tool)

            # Add get meetings tool
            get_meetings_tool = GetMeetingsTool()
            await get_meetings_tool.initialize()
            self.add_tool(get_meetings_tool)

            get_query_tool = GetQueryTool()
            await get_query_tool.initialize()
            self.add_tool(get_query_tool)

            get_content_tool = GetContentTool()
            await get_content_tool.initialize()
            self.add_tool(get_content_tool)

        # Initialize Gemini client
        logger.info("Connecting to Gemini API...")
        self.client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=self.api_key)

        # Get tools configuration
        tools = self._create_tools_config()

        # Configure the Gemini model with enhanced system prompt and tools
        self.config = types.LiveConnectConfig(
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=self.system_prompt)],
                role="user"
            ),
            tools=tools
        )

        logger.info("Initialization complete!")
        return True

    async def run(self):
        """Run the agent"""
        logger.info(f"{self.name} with Gemini Live API starting...")

        try:
            # Connect to Gemini API
            async with (
                self.client.aio.live.connect(model=self.model_name, config=self.config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                logger.info("Connected to Gemini API successfully!")

                # Create all the async tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.voice_capture.listen_audio(
                    self.out_queue,
                    self.is_wakeword_active,
                    self.is_wakeword_detected,
                    self.update_last_activity,
                    self.is_speaking,
                    self.set_wakeword_active
                ))
                tg.create_task(self.check_for_return_to_wakeword())
                tg.create_task(self.check_sleepword_detector())

                # Initialize video capture based on mode
                if self.video_mode == "camera":
                    camera_tool = self.get_tool("CameraCapture")
                    if camera_tool:
                        tg.create_task(self.get_camera_frames(camera_tool))
                elif self.video_mode == "screen":
                    screen_tool = self.get_tool("ScreenCapture")
                    if screen_tool:
                        tg.create_task(self.get_screen_frames(screen_tool))

                tg.create_task(self.receive_audio())
                tg.create_task(self.voice_capture.play_audio(self.audio_in_queue))

                logger.info(f"{self.name} ready! Say the wake word to begin.")
                print(f"\n{self.name} ready! Say the wake word to begin.")
                print("Type 'q' to quit or 'listen' to return to wake word mode.")

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            logger.info(f"Shutting down {self.name}...")
        except Exception as e:
            logger.error(f"Error in agent execution: {e}", exc_info=True)
        finally:
            await self.cleanup()
            logger.info(f"{self.name} shutdown complete.")

    async def handle_tool_call(self, tool_call):
        """Handle tool calls from Gemini API"""
        function_responses_list = []

        for fc in tool_call.function_calls:
            function_name = fc.name
            args = dict(fc.args)
            response_data = None

            logger.info(f"Received request to call function: {function_name}")
            logger.info(f"Arguments provided by Gemini: {args}")

            try:
                if function_name == "schedule_google_meet":
                    # Call the schedule meeting tool
                    schedule_meeting_tool = self.get_tool("ScheduleMeeting")
                    if not schedule_meeting_tool:
                        raise ValueError("ScheduleMeeting tool not available")

                    response_data = await schedule_meeting_tool.execute(**args)

                elif function_name == "get_scheduled_meetings":
                    # Call the get meetings tool
                    get_meetings_tool = self.get_tool("GetMeetings")
                    if not get_meetings_tool:
                        raise ValueError("GetMeetings tool not available")

                    response_data = await get_meetings_tool.execute(**args)

                elif function_name == "query_docs":
                    get_query_tool = self.get_tool("GetQuery")
                    if not get_query_tool:
                        raise ValueError("GetQuery tool not available")

                    response_data = await get_query_tool.execute(**args)

                elif function_name == "generate_meeting_content":
                    get_content_tool = self.get_tool("GetContent")
                    if not get_content_tool:
                        raise ValueError("GetContent tool not available")

                    response_data = await get_content_tool.execute(**args)


                else:
                    # Handle unexpected function call requests
                    error_msg = f"Function '{function_name}' is not supported"
                    logger.warning(error_msg)
                    response_data = {"status": "error", "message": error_msg}

            except Exception as e:
                # Catch errors during argument validation or function execution
                error_msg = f"Error executing function '{function_name}': {e}"
                logger.error(error_msg, exc_info=True)
                response_data = {"status": "error", "message": error_msg}

            # Append the function response
            function_responses_list.append(types.FunctionResponse(
                name=function_name,
                id=fc.id,
                response=response_data
            ))
            logger.info(f"Result for {function_name}: {response_data}")

        # Send all collected responses back to Gemini
        tool_response = types.LiveClientToolResponse(
            function_responses=function_responses_list
        )

        logger.info("Sending tool response back to Gemini")
        await self.session.send(input=tool_response)

    async def cleanup(self):
        """Clean up resources"""
        # Clean up wake word detector
        if self.wakeword_detector:
            self.wakeword_detector.stop()
        # Clean up sleep word detector
        if self.sleepword_detector:
            self.sleepword_detector.stop()
        # Clean up voice capture
        if self.voice_capture:
            self.voice_capture.cleanup()
        # Clean up tools
        for tool_name, tool in self.tools.items():
            if hasattr(tool, 'cleanup'):
                tool.cleanup()

    # Helper methods
    def is_wakeword_active(self):
        """Check if wake word detection is active"""
        return self.wakeword_active

    def is_wakeword_detected(self):
        """Check if wake word has been detected"""
        return self.wakeword_detector.detected

    def update_last_activity(self):
        """Update the last activity timestamp"""
        self.last_activity_time = time.time()

    def is_speaking(self):
        """Check if the agent is currently speaking"""
        return self.speaking

    async def set_wakeword_active(self, active):
        """Set whether wake word detection is active"""
        self.wakeword_active = active
        if active:
            self.wakeword_detector.reset()
            logger.info("Switched to wake word listening mode")

    # Agent tasks
    async def send_text(self):
        """Send text input to Gemini (for debugging and testing)"""
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            if text.lower() == "listen":
                await self.switch_to_wakeword_mode("Returning to wake word mode.")
                continue

            # Handle special commands for calendar tools
            if text.lower().startswith("!schedule"):
                # Example: !schedule 2025-04-20 15:00:00 "Meeting Title" "Description" 60 user1@example.com,user2@example.com UTC
                # Parse the command
                try:
                    parts = text[9:].strip().split(' ', 3)  # Split first 3 spaces
                    date_str = parts[0]
                    time_str = parts[1]

                    # Extract remaining parts
                    remaining = parts[2:]
                    if len(remaining) > 0:
                        # The rest might have quoted strings and comma-separated lists
                        import shlex
                        args = shlex.split(' '.join(remaining))

                        summary = args[0] if len(args) > 0 else "Meeting"
                        description = args[1] if len(args) > 1 else ""
                        duration = int(args[2]) if len(args) > 2 else 60

                        attendees = None
                        if len(args) > 3:
                            attendees = [email.strip() for email in args[3].split(',')]

                        timezone = args[4] if len(args) > 4 else "UTC"

                        logger.info(f"Scheduling meeting: {summary} on {date_str} at {time_str}")
                        result = await self.schedule_meeting(
                            date_str=date_str,
                            time_str=time_str,
                            summary=summary,
                            description=description,
                            duration_minutes=duration,
                            attendees=attendees,
                            timezone=timezone
                        )

                        print(result)
                        continue
                except Exception as e:
                    logger.error(f"Error parsing schedule command: {e}", exc_info=True)
                    print(f"Error parsing schedule command: {e}")
                    print("Usage: !schedule YYYY-MM-DD HH:MM:SS \"Title\" \"Description\" duration_minutes emails,comma,separated timezone")
                    continue

            elif text.lower().startswith("!meetings"):
                # Example: !meetings 2025-04-16 00:00:00 2025-04-23 23:59:59 UTC
                try:
                    parts = text[9:].strip().split()
                    if len(parts) >= 4:
                        start_date = parts[0]
                        start_time = parts[1]
                        end_date = parts[2]
                        end_time = parts[3]
                        timezone = parts[4] if len(parts) > 4 else "UTC"

                        logger.info(f"Getting meetings from {start_date} {start_time} to {end_date} {end_time} ({timezone})")
                        result = await self.get_meetings(
                            start_date_str=start_date,
                            start_time_str=start_time,
                            end_date_str=end_date,
                            end_time_str=end_time,
                            timezone=timezone
                        )

                        print(result)
                        continue
                except Exception as e:
                    logger.error(f"Error parsing meetings command: {e}", exc_info=True)
                    print(f"Error parsing meetings command: {e}")
                    print("Usage: !meetings start_date start_time end_date end_time timezone")
                    continue

            # Regular message
            await self.session.send(input=text or ".", end_of_turn=True)
            self.update_last_activity()

            # Add to conversation context
            self.conversation_context.append({"role": "user", "content": text})

    async def send_realtime(self):
        """Send data to Gemini in real-time"""
        while True:
            msg = await self.out_queue.get()
            if not self.wakeword_active:  # Only send to Gemini when not in wake word mode
                if "text" in msg:
                    await self.session.send(input=msg["text"], end_of_turn=True)
                else:
                    await self.session.send(input=msg)
                    # Update activity timestamp for audio messages
                    if msg.get("mime_type") == "audio/pcm":
                        self.update_last_activity()

    async def receive_audio(self):
        """Background task to read from the websocket and write pcm chunks to the output queue"""
        while True:
            turn = self.session.receive()
            self.speaking = True
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                    # Add to conversation context for continuity
                    if text.strip():
                        self.conversation_context.append({"role": "assistant", "content": text})

                # Handle tool calls
                if tool_call := response.tool_call:
                    await self.handle_tool_call(tool_call)

            print()  # Add a newline after each response for readability
            self.speaking = False
            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def switch_to_wakeword_mode(self, message=None):
        """Switch to wake word mode with an optional goodbye message"""
        # if message and not self.wakeword_active:
        #     await self.session.send(input=message, end_of_turn=True)
        #     # Allow time for the message to be processed and spoken
        #     await asyncio.sleep(1)

        await self.set_wakeword_active(True)

    async def check_for_return_to_wakeword(self):
        """Check for timeout to return to wake word mode"""
        while True:
            await asyncio.sleep(1)

            if not self.wakeword_active and not self.speaking:
                # Check if timeout has been reached
                current_time = time.time()
                if current_time - self.last_activity_time > self.inactivity_timeout:
                    await self.switch_to_wakeword_mode("I'll be here if you need anything else.")
            else:
                self.update_last_activity()  # Reset timer when in wake word mode

    async def check_sleepword_detector(self):
        """Monitor the sleep word detector and switch modes when detected"""
        while True:
            await asyncio.sleep(0.1)

            # Only check for sleep word when not in wake word mode
            if not self.wakeword_active and self.sleepword_detector and self.sleepword_detector.detected:
                logger.info("Sleep word detected, switching to wake word mode...")
                await self.switch_to_wakeword_mode("Muted. Call me again when you need me.")
                self.sleepword_detector.reset()

    async def get_camera_frames(self, camera_tool):
        """Capture and send frames from camera"""
        while True:
            if not self.wakeword_active:  # Only send frames when actively listening
                frame = await camera_tool.execute()
                if frame is None:
                    await asyncio.sleep(0.5)
                    continue

                await asyncio.sleep(1.0)
                await self.out_queue.put(frame)
            else:
                await asyncio.sleep(0.5)  # Sleep while in wake word mode

    async def get_screen_frames(self, screen_tool):
        """Capture and send screen frames"""
        while True:
            if not self.wakeword_active:  # Only send screen when actively listening
                frame = await screen_tool.execute()
                if frame is None:
                    await asyncio.sleep(0.5)
                    continue

                await asyncio.sleep(1.0)
                await self.out_queue.put(frame)
            else:
                await asyncio.sleep(0.5)  # Sleep while in wake word mode

    # Calendar tool methods
    async def schedule_meeting(self, date_str, time_str, summary, description="",
                             duration_minutes=60, attendees=None, timezone='UTC'):
        """Schedule a meeting with Google Calendar"""
        schedule_meeting_tool = self.get_tool("ScheduleMeeting")
        if not schedule_meeting_tool:
            return {"status": "error", "message": "ScheduleMeeting tool not available"}

        return await schedule_meeting_tool.execute(
            date_str=date_str,
            time_str=time_str,
            summary=summary,
            description=description,
            duration_minutes=duration_minutes,
            attendees=attendees,
            timezone=timezone
        )

    async def get_meetings(self, start_date_str, start_time_str, end_date_str, end_time_str,
                         timezone='UTC', calendar_id='primary', max_results=50):
        """Get meetings from Google Calendar"""
        get_meetings_tool = self.get_tool("GetMeetings")
        if not get_meetings_tool:
            return {"status": "error", "message": "GetMeetings tool not available"}

        return await get_meetings_tool.execute(
            start_date_str=start_date_str,
            start_time_str=start_time_str,
            end_date_str=end_date_str,
            end_time_str=end_time_str,
            timezone=timezone,
            calendar_id=calendar_id,
            max_results=max_results
        )