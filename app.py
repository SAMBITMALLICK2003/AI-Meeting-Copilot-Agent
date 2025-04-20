import os
import datetime
import traceback
import asyncio
import sys
import threading
import speechmatics.client
from Agent_Framework.transcription.transcription_help import (
    FileWriterTranscripts,
    add_printing_handlers,
    transcribe_from_speaker,
)
from Agent_Framework.agent.live_agent import LiveAgent
from Agent_Framework.config.settings import speechmatics_auth_token, transcription_file_path, gemini_embedding_model, llm

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# Define paths
TRANSCRIPTION_FILE = "meeting_storage/transcription.txt"
OUTPUT_FILE = "output_storage/output.txt"

# Ensure directories exist
os.makedirs("meeting_storage", exist_ok=True)
os.makedirs("output_storage", exist_ok=True)

# Create empty files if they don't exist
if not os.path.exists(TRANSCRIPTION_FILE):
    with open(TRANSCRIPTION_FILE, 'w', encoding='utf-8') as f:
        f.write("")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("")

# Import UI components
from PyQt5.QtWidgets import QApplication
from meeting_copilot_ui import MeetingCopilotUI

import shutil
import os


def delete_folder(folder_path):
    """
    Delete a folder and all its contents.

    Args:
        folder_path (str): Path to the folder to delete

    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        if os.path.exists(folder_path):
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"Successfully deleted folder: {folder_path}")
                return True
            else:
                print(f"Error: {folder_path} is not a directory")
                return False
        else:
            print(f"Folder does not exist: {folder_path}")
            return False
    except Exception as e:
        print(f"Error deleting folder {folder_path}: {str(e)}")
        return False

def build_index(doc_path="./RAG_Database"):
    # check if storage already exists
    Settings.llm = llm
    Settings.embed_model = gemini_embedding_model
    PERSIST_DIR = "./storage"
    # Example usage:
    if delete_folder("./storage"):
        print("Vector database storage deleted.")
    if not os.path.exists(PERSIST_DIR):
        # load the documents and create the index
        documents = SimpleDirectoryReader(doc_path).load_data()

        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

index = build_index()

async def run_transcription_background():
    word_list = []
    live_buffer = []
    live_doc_ids = []
    auth_token = speechmatics_auth_token
    language = "en"
    max_delay = 2.0
    output_file = transcription_file_path

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_file:
        output_file = f"transcription_{timestamp}.txt"

    abs_path = os.path.abspath(output_file)
    print(f"[Transcription] Will save transcription to: {abs_path}")

    transcripts = FileWriterTranscripts(text="", json=[], output_file=output_file)
    speechmatics_client = speechmatics.client.WebsocketClient(connection_settings_or_auth_token=auth_token)
    add_printing_handlers(speechmatics_client, transcripts, word_list, live_buffer, live_doc_ids)

    try:
        await transcribe_from_speaker(speechmatics_client, language, max_delay)
    except asyncio.CancelledError:
        print("[Transcription] Cancelled.")
    except Exception as e:
        print(f"[Transcription] Error: {e}")
        traceback.print_exc()
    finally:
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"\n===== Transcription ended at {timestamp} =====\n")
            print(f"[Transcription] Final transcription saved to: {abs_path}")
        except Exception as e:
            print(f"[Transcription] Error writing final footer: {e}")
            traceback.print_exc()


async def run_calendar_agent():
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    agent = LiveAgent(
        name="Calendar Assistant",
        system_prompt=(
            f"""You are Alexa, a sophisticated AI assistant designed to be helpful, engaging, and personable. 
            Your personality is friendly, conversational, and responsive.

            Today's date is {today_date}. If you are told to convert today or tomorrow's date to yyyy-mm-dd format, 
            you should calculate it on your own. Always speak in English. If a task requires the use of a function call, 
            explicitly mention what you are about to do before initiating the function call. Once the function call is 
            complete, clearly explain what task has been completed automatically as a result of the function call.

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
            - Generating meeting summaries, minutes, and action items
            - Searching through uploaded documents

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

            Remember to be helpful, harmless, and honest in all interactions.

            IMPORTANT: When asked to generate a meeting summary, minutes, or action items, make sure to use the transcription
            data to create relevant and accurate information. Your generated content should be written to the output file
            which will then be displayed in the UI's Generated Content tab."""
        ),
        video_mode="screen",
        enable_calendar=True
    )

    await agent.initialize()
    await agent.run()


async def main_async():
    # Start transcription and calendar assistant concurrently
    transcription_task = asyncio.create_task(run_transcription_background())  # Transcription runs in the background
    calendar_task = asyncio.create_task(run_calendar_agent())  # Calendar agent runs in the foreground

    try:
        # Wait for both tasks to complete concurrently
        await asyncio.gather(calendar_task, transcription_task)
    finally:
        # Cancel transcription if calendar agent exits
        transcription_task.cancel()
        await transcription_task


def start_backend():
    """Run the backend in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main_async())


def main():
    """Main function that runs both the UI and backend"""
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend)
    backend_thread.daemon = True  # Thread will exit when main program exits
    backend_thread.start()

    # Start the UI in the main thread
    app = QApplication(sys.argv)
    window = MeetingCopilotUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted.")