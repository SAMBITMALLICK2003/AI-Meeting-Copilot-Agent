"""
Google Calendar integration tools
Current Date and Time (UTC): 2025-04-16 17:44:46
Current User's Login: SAMBITMALLICK2003
"""
from pathlib import Path
import logging
from google import genai
from google.genai import types
from ..tools.base_tool import BaseTool
from ..config.settings import API_KEY, CONTENT_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ContentTool(BaseTool):
    """Base class for Query Tool"""

    def __init__(self, name, description):
        super().__init__(name, description)
        self.service = None


class GetContentTool(ContentTool):
    """Tool for generate content"""

    def __init__(self):
        super().__init__(
            name="GetContent",
            description="Generates different types of meeting content such as summary, minutes, or action items based on the provided query."
        )


    async def execute(self, query):
        if not self.service:
            if not await self.initialize():
                return {
                    "status": "error",
                    "message": "Failed to get the query service"
                }

        try:
            text = ""
            client = genai.Client(
                api_key=API_KEY,
            )

            model = CONTENT_MODEL

            with open(str(Path(__file__).resolve().parent.parent.parent / "meeting_storage" / "transcription.txt"), 'r', encoding='utf-8') as file:
                raw_content = file.read()
                # Filter out lines starting with "Alexa" and ending with "mute"
                import re
                content = re.sub(r'alexa.*?mute', '', raw_content, flags=re.IGNORECASE)
                print(f"Filtered content: {content}")
            compiled_query = f"{content}\n\nUser Request: {query}"
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=compiled_query),
                    ],
                ),
            ]
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                system_instruction=[
                    types.Part.from_text(text="""You are an AI assistant specialized in analyzing and summarizing meeting transcriptions with precision and accuracy. Your task is to extract meaningful information from the provided meeting transcription while adhering to the following detailed guidelines. While giving the answer no need to mention when that transcription was generated. Only give answer to specific query."""),
                ],
            )

            for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
            ):
                text += chunk.text + " "

            with open(str(Path(__file__).resolve().parent.parent.parent / "output_storage" / "output.txt"), 'w', encoding='utf-8') as out_file:
                out_file.write(text)

            return {
                "status": "success",
                "Content response": text
            }

        except Exception as e:
            logger.error(f"Error during document query: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }


