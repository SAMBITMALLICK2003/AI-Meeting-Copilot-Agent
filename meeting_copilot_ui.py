import sys
import os
import asyncio
import threading
import time
import json
from google import genai
from google.genai import types
from Agent_Framework.config.settings import API_KEY, CONTENT_MODEL, CHAT_API_KEY, GROQ_API_KEY
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QComboBox,
    QFrame, QSplitter, QProgressBar, QMessageBox, QStatusBar,
    QCheckBox, QTabWidget, QScrollArea, QSizePolicy, QGraphicsDropShadowEffect,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QIcon, QFont, QColor, QPalette, QTextCursor, QLinearGradient, QBrush, QPainter, QPixmap, \
    QFontDatabase
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from llama_index.core import StorageContext, load_index_from_storage
from groq import Groq


# Define the path for transcription file
TRANSCRIPTION_FILE = Path("meeting_storage/transcription.txt")
OUTPUT_FILE = Path("output_storage/output.txt")
PERSIST_DIR = "./storage"

# Constants for fact detection
FACT_TRIGGER_THRESHOLD = 0.75  # Threshold for similarity matching

# Phrases that indicate numerical facts or important information
FACT_TRIGGER_PHRASES =  [
    # General price phrases
    "Can you confirm the final price ",
    "the price of",
    "it costs",
    "this is priced at",
    "how much is",
    "what is the rate of",
    "current rate of",
    "value of",
    "selling price",
    "retail price",
    "market price",

    # Tax-related phrases
    "including taxes",
    "excluding taxes",
    "with tax",
    "before tax",
    "after tax",

    # Weight-related phrases
    "per gram",
    "for 1 gram",
    "price for 10 grams",
    "denomination",
    "how much for",
    "X gram costs",
    "price in INR",

    # Product-specific references (from the document)
    "lotus ingot",
    "certicard",
    "minted bar",
    "laxmi round",
    "rose oval coin",
    "stylized laxmi ganesh",
    "casted bar",
    "banyan tree ingot",
    "shankh ganesh laxmi set",
    "peacock ingot",
    "ram lalla minted card",
    "ganesh laxmi coin",
    "guru nanak dev minted bar",
    "ashta laxmi coin",
    "sukh samridhi colour",
    "gift for newborn baby",
    "raksha bandhan coin",

    # Buyback references
    "buyback price",
    "buyback rate",
    "resell value",
    "return price"
]

# Define a color scheme
COLORS = {
    "primary": "#3498db",  # Blue
    "secondary": "#2ecc71",  # Green
    "accent": "#e74c3c",  # Red
    "light": "#ecf0f1",  # Light gray
    "dark": "#2c3e50",  # Dark blue/gray
    "warning": "#f39c12",  # Orange
    "success": "#27ae60",  # Dark green
    "dark_mode_bg": "#1e272e",  # Dark background
    "dark_mode_text": "#d2dae2"  # Light text for dark mode
}


class GradientTitleBar(QWidget):
    """Custom title bar with gradient background"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setMaximumHeight(70)

        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 5, 15, 5)

        # Create title label
        self.title_label = QLabel("Meeting Copilot")
        font = QFont("Arial", 20, QFont.Bold)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet(f"color: white; padding: 5px;")

        # Create subtitle
        self.subtitle = QLabel("AI-Powered Meeting Assistant")
        subtitle_font = QFont("Arial", 10, QFont.Normal)
        self.subtitle.setFont(subtitle_font)
        self.subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.8);")

        # Create vertical layout for title and subtitle
        title_layout = QVBoxLayout()
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle)

        # Add layouts to main layout
        layout.addLayout(title_layout)
        layout.addStretch()

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)

    def paintEvent(self, event):
        """Paint the gradient background"""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, QColor(COLORS["primary"]))
        gradient.setColorAt(1, QColor(COLORS["dark"]))
        painter.fillRect(self.rect(), QBrush(gradient))


class StyledButton(QPushButton):
    """Custom styled button with animations"""

    def __init__(self, text, parent=None, primary=True):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(36)

        # Set up colors
        if primary:
            self.base_color = COLORS["primary"]
            self.hover_color = "#2980b9"  # Darker blue
            self.text_color = "white"
        else:
            self.base_color = COLORS["light"]
            self.hover_color = "#bdc3c7"  # Darker gray
            self.text_color = COLORS["dark"]

        # Set default style
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.base_color};
                color: {self.text_color};
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.hover_color};
            }}
            QPushButton:pressed {{
                background-color: {self.hover_color};
                padding: 9px 15px 7px 17px;
            }}
        """)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)


class AsyncSignals(QObject):
    """Class to emit signals from async functions to the Qt main thread"""
    agent_status_changed = pyqtSignal(str)
    transcription_updated = pyqtSignal(str)
    output_updated = pyqtSignal(str)


class AsyncChatSignals(QObject):
    """Class to emit signals from async functions to the Qt main thread"""
    message_received = pyqtSignal(str, str)  # sender, message
    fact_detected = pyqtSignal(str, str)  # trigger phrase, relevant info
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()


class StyledTabWidget(QTabWidget):
    """Custom styled tab widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }

            QTabBar::tab {
                background-color: #f0f0f0;
                color: #555;
                border: 1px solid #ddd;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                min-width: 150px;
                font-weight: normal;
            }

            QTabBar::tab:selected {
                background-color: white;
                color: #3498db;
                font-weight: bold;
                border-bottom: 2px solid #3498db;
            }

            QTabBar::tab:hover:!selected {
                background-color: #e0e0e0;
            }
        """)


class MessageWidget(QFrame):
    """Custom widget to display a chat message"""

    def __init__(self, message, is_user=True, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(1)

        # Set different styles for user and AI messages
        if is_user:
            self.setStyleSheet("""
                QFrame {
                    background-color: #e1f5fe;
                    border-radius: 10px;
                    border: 1px solid #b3e5fc;
                    margin: 5px;
                    padding: 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: #f1f1f1;
                    border-radius: 10px;
                    border: 1px solid #e0e0e0;
                    margin: 5px;
                    padding: 5px;
                }
            """)

        # Create layout
        layout = QVBoxLayout(self)

        # Create header with sender name and timestamp
        header_layout = QHBoxLayout()
        sender_label = QLabel("You" if is_user else "AI Assistant")
        sender_label.setStyleSheet("font-weight: bold;")

        timestamp = datetime.now().strftime("%H:%M:%S")
        time_label = QLabel(timestamp)
        time_label.setStyleSheet("color: gray; font-size: 10px;")

        header_layout.addWidget(sender_label)
        header_layout.addStretch()
        header_layout.addWidget(time_label)

        # Create message text
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Add to layout
        layout.addLayout(header_layout)
        layout.addWidget(message_label)


class FactWidget(QFrame):
    """Custom widget to display automatic fact findings"""

    def __init__(self, trigger, fact, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setLineWidth(1)

        # Set style for fact widgets
        self.setStyleSheet("""
            QFrame {
                background-color: #fff8e1;
                border-radius: 10px;
                border: 1px solid #ffe082;
                margin: 5px;
                padding: 5px;
            }
        """)

        # Create layout
        layout = QVBoxLayout(self)

        # Create header
        header_layout = QHBoxLayout()
        info_icon = QLabel()
        # In real implementation, load an actual icon
        # info_icon.setPixmap(QPixmap("info_icon.png").scaled(16, 16))

        header_label = QLabel("Relevant Information")
        header_label.setStyleSheet("font-weight: bold; color: #f57c00;")

        header_layout.addWidget(info_icon)
        header_layout.addWidget(header_label)
        header_layout.addStretch()

        # Create trigger phrase text
        trigger_label = QLabel(f"Related to: \"{trigger}\"")
        trigger_label.setStyleSheet("font-style: italic; color: #7f7f7f;")

        # Create fact text
        fact_label = QLabel(fact)
        fact_label.setWordWrap(True)
        fact_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Add to layout
        layout.addLayout(header_layout)
        layout.addWidget(trigger_label)
        layout.addWidget(fact_label)


class ChatWithAITab(QWidget):
    """Tab for chatting with AI and viewing automatically detected facts"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = AsyncChatSignals()
        self.last_processed_transcription = ""
        self.sentence_transformer = None
        self.fact_trigger_embeddings = None
        self.rag_index = None

        # Initialize UI
        self.init_ui()

        # Connect signals
        self.signals.message_received.connect(self.add_message)
        self.signals.fact_detected.connect(self.add_fact)
        self.signals.processing_started.connect(self.on_processing_started)
        self.signals.processing_finished.connect(self.on_processing_finished)

        # Load necessary models and indexes in the background
        self.load_models_thread = threading.Thread(target=self.load_models)
        self.load_models_thread.daemon = True
        self.load_models_thread.start()

        # Set up timer for processing transcriptions
        self.transcription_timer = QTimer(self)
        self.transcription_timer.timeout.connect(self.process_transcription)
        self.transcription_timer.start(5000)  # Check every 5 seconds

    def load_models(self):
        """Load sentence transformer model and RAG index"""
        try:
            print("Loading sentence transformer model...")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

            # Precompute embeddings for trigger phrases
            self.fact_trigger_embeddings = {}
            for phrase in FACT_TRIGGER_PHRASES:
                self.fact_trigger_embeddings[phrase] = self.sentence_transformer.encode(phrase)

            print("Loading RAG index...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                self.rag_index = load_index_from_storage(storage_context)
                print("RAG index loaded successfully")
            except Exception as e:
                print(f"Error loading RAG index: {e}")

            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")

    def init_ui(self):
        """Initialize the UI components"""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Splitter to allow resizing between chat and facts
        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Chat section (left side)
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        # Chat history
        self.chat_scroll_area = QScrollArea()
        self.chat_scroll_area.setWidgetResizable(True)
        self.chat_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chat_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.chat_scroll_area.setWidget(self.chat_container)

        # Message input and send button
        input_layout = QHBoxLayout()

        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Type your message here...")
        self.message_input.setMaximumHeight(80)
        self.message_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 8px;
                background-color: white;
            }
        """)

        self.send_button = QPushButton("Send")
        self.send_button.setMinimumHeight(80)
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2980b9;
                padding: 9px 15px 7px 17px;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.message_input, 7)
        input_layout.addWidget(self.send_button, 1)

        # Add widgets to chat layout
        chat_layout.addWidget(QLabel("Chat with AI"))
        chat_layout.addWidget(self.chat_scroll_area)
        chat_layout.addLayout(input_layout)

        # Facts section (right side)
        facts_widget = QWidget()
        facts_layout = QVBoxLayout(facts_widget)
        facts_layout.setContentsMargins(0, 0, 0, 0)

        # Facts list
        facts_label = QLabel("Automatic Fact Detection")
        facts_label.setStyleSheet("font-weight: bold;")

        self.facts_scroll_area = QScrollArea()
        self.facts_scroll_area.setWidgetResizable(True)
        self.facts_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.facts_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.facts_container = QWidget()
        self.facts_layout = QVBoxLayout(self.facts_container)
        self.facts_layout.setAlignment(Qt.AlignTop)
        self.facts_layout.setSpacing(10)
        self.facts_scroll_area.setWidget(self.facts_container)

        facts_layout.addWidget(facts_label)
        facts_layout.addWidget(self.facts_scroll_area)

        # Add to splitter
        splitter.addWidget(chat_widget)
        splitter.addWidget(facts_widget)

        # Set initial sizes (70% chat, 30% facts)
        splitter.setSizes([700, 300])

        # Add to main layout
        main_layout.addWidget(splitter)

        # Welcome message
        welcome_widget = MessageWidget(
            "Hello! I'm your AI assistant. I can help answer questions about your meeting and documents. "
            "I'll also automatically highlight important facts and figures from the ongoing conversation.",
            is_user=False
        )
        self.chat_layout.addWidget(welcome_widget)

    def send_message(self):
        """Send a message to the AI and display it in the chat"""
        message = self.message_input.toPlainText().strip()
        if not message:
            return

        # Add user message to chat
        self.add_message("You", message)

        # Clear input field
        self.message_input.clear()

        # Process with AI in a separate thread
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()

    def process_message(self, message):
        """Process the user's message with Gemini API and RAG"""
        self.signals.processing_started.emit()

        try:
            # client = genai.Client(api_key=CHAT_API_KEY)
            client = Groq(api_key=GROQ_API_KEY)

            # First, try to find relevant information from RAG
            rag_response = self.query_rag(message)

            # Create system instruction
            system_instruction = """You are an AI assistant specialized in helping with meetings.
            You're friendly, helpful, and concise. When responding:
            1. If relevant information from documents is provided, use it to enhance your answer
            2. Be specific and detailed when answering questions
            3. If you don't know the answer, say so honestly
            4. Format your responses with markdown for better readability
            5. Keep responses concise but informative
            6. If the information is not in RAG, tell that the information is not found"""

            # Create context with transcription
            context = ""
            if TRANSCRIPTION_FILE.exists():
                try:
                    with open(TRANSCRIPTION_FILE, 'r', encoding='utf-8') as f:
                        # Get the last 2000 characters of transcription for context
                        full_text = f.read()
                        if len(full_text) > 2000:
                            context = "..." + full_text[-2000:]
                        else:
                            context = full_text
                except Exception as e:
                    print(f"Error reading transcription file: {e}")

            # Create prompt with context and RAG information
            prompt = f"""User question: {message}

Context from meeting transcription:
{context}

"""
            if rag_response and rag_response.strip():
                prompt += f"""
Relevant information from documents:
{rag_response}
"""

            # Build the messages structure
            messages = [
                {
                    "role": "system",
                    "content": system_instruction
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Generate content with ChatGROQ
            response = client.chat.completions.create(
                model="llama3-70b-8192",  # Use the appropriate ChatGROQ model
                messages=messages,
                temperature=0.1,  # More deterministic, similar to your original setting
            )

            # Get the response text
            response_text = response.choices[0].message.content

            # Send to chat
            self.signals.message_received.emit("AI Assistant", response_text)

        except Exception as e:
            print(f"Error processing message: {e}")
            self.signals.message_received.emit("AI Assistant",
                                               f"I'm sorry, I encountered an error while processing your request: {str(e)}")

        self.signals.processing_finished.emit()

    def query_rag(self, query):
        """Query the RAG system for relevant information"""
        if not self.rag_index:
            print("RAG index not loaded yet")
            return None

        try:
            # Create query engine
            query_engine = self.rag_index.as_query_engine()

            # Execute query
            response = query_engine.query(query)

            return str(response)

        except Exception as e:
            print(f"Error querying RAG: {e}")
            return None

    def process_transcription(self):
        """Process new transcription text to detect facts and important information"""
        if not TRANSCRIPTION_FILE.exists() or not self.sentence_transformer or not self.rag_index:
            return

        try:
            with open(TRANSCRIPTION_FILE, 'r', encoding='utf-8') as f:
                current_transcription = f.read()

            # Check if there's new content
            if current_transcription == self.last_processed_transcription:
                return

            # Find new content
            if self.last_processed_transcription and len(self.last_processed_transcription) < len(
                    current_transcription):
                new_text = current_transcription[len(self.last_processed_transcription):]
            else:
                new_text = current_transcription

            # Update last processed transcription
            self.last_processed_transcription = current_transcription

            # Process new text if it exists
            if new_text and len(new_text) > 10:  # At least a few words
                self.analyze_text_for_facts(new_text)

        except Exception as e:
            print(f"Error processing transcription: {e}")

    def analyze_text_for_facts(self, text):
        """Analyze text to find fact triggers and query RAG"""
        if not self.sentence_transformer or not self.rag_index:
            return

        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            if len(sentence) < 10:  # Skip very short sentences
                continue

            # Encode sentence
            sentence_embedding = self.sentence_transformer.encode(sentence)

            # Compare with trigger phrases
            for phrase, phrase_embedding in self.fact_trigger_embeddings.items():
                # Calculate cosine similarity
                similarity = np.dot(sentence_embedding, phrase_embedding) / (
                        np.linalg.norm(sentence_embedding) * np.linalg.norm(phrase_embedding)
                )

                # If similarity is above threshold, consider it a match
                if similarity > FACT_TRIGGER_THRESHOLD:
                    print(f"Fact trigger detected: '{phrase}' in '{sentence}'")
                    self.find_relevant_information(sentence, phrase)
                    break  # Only trigger once per sentence

    def find_relevant_information(self, trigger_sentence, trigger_phrase):
        """Find relevant information in RAG based on trigger"""
        if not self.rag_index:
            return

        try:
            # Create query engine
            query_engine = self.rag_index.as_query_engine()

            # Format query to extract facts related to the trigger
            query = f"Find information related to: {trigger_sentence}"

            # Execute query
            response = query_engine.query(query)
            response_text = str(response)

            # If meaningful response found, show it
            if response_text and len(response_text) > 20 and "no relevant" not in response_text.lower():
                self.signals.fact_detected.emit(trigger_sentence, response_text)

        except Exception as e:
            print(f"Error finding relevant information: {e}")

    def add_message(self, sender, message):
        """Add a message to the chat display"""
        is_user = sender == "You"
        message_widget = MessageWidget(message, is_user=is_user)
        self.chat_layout.addWidget(message_widget)

        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_chat_to_bottom)

    def add_fact(self, trigger, fact):
        """Add an automatically detected fact to the facts display"""
        fact_widget = FactWidget(trigger, fact)
        self.facts_layout.addWidget(fact_widget)

        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_facts_to_bottom)

    def scroll_chat_to_bottom(self):
        """Scroll the chat view to the bottom"""
        self.chat_scroll_area.verticalScrollBar().setValue(
            self.chat_scroll_area.verticalScrollBar().maximum()
        )

    def scroll_facts_to_bottom(self):
        """Scroll the facts view to the bottom"""
        self.facts_scroll_area.verticalScrollBar().setValue(
            self.facts_scroll_area.verticalScrollBar().maximum()
        )

    def on_processing_started(self):
        """Handle UI updates when processing starts"""
        self.send_button.setEnabled(False)
        self.send_button.setText("Processing...")

    def on_processing_finished(self):
        """Handle UI updates when processing finishes"""
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")


class MeetingCopilotUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up window properties
        self.setWindowTitle("Meeting Copilot")
        self.setMinimumSize(1000, 700)

        # Load custom fonts
        self.load_custom_fonts()

        # State variables
        self.agent_running = False
        self.signals = AsyncSignals()
        self.transcription_last_size = 0
        self.output_last_size = 0
        self.last_output_content = ""
        self.socket_path = "/tmp/gemini_agent_socket"  # Socket for communicating with agent
        self.is_dark_mode = False

        # Set up UI
        self.init_ui()

        # Connect signals
        self.signals.agent_status_changed.connect(self.update_agent_status)
        self.signals.transcription_updated.connect(self.update_transcription)
        self.signals.output_updated.connect(self.update_output)

        # Set up timers for file monitoring
        self.setup_file_monitoring()

        # Create directories if they don't exist
        os.makedirs("meeting_storage", exist_ok=True)
        os.makedirs("output_storage", exist_ok=True)

    def load_custom_fonts(self):
        """Load custom fonts for the application"""
        # In a real application, you would add custom font files to your resources
        # and load them here. For this example, we'll use system fonts.
        pass

    def init_ui(self):
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for full-width title bar

        # Add custom title bar
        self.title_bar = GradientTitleBar(self)
        main_layout.addWidget(self.title_bar)

        # Container for content below title bar
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.addWidget(content_container)

        # Create a tab widget with custom styling
        self.tabs = StyledTabWidget()
        content_layout.addWidget(self.tabs)

        # Create tab for transcription
        transcription_tab = QWidget()
        transcription_layout = QVBoxLayout(transcription_tab)

        # Transcription header
        transcription_header = QHBoxLayout()
        transcription_label = QLabel("Live Transcription")
        transcription_label.setFont(QFont("Arial", 14, QFont.Bold))
        transcription_label.setStyleSheet(f"color: {COLORS['primary']};")
        transcription_header.addWidget(transcription_label)
        transcription_layout.addLayout(transcription_header)

        # Transcription text area with styled border
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setFont(QFont("Arial", 11))
        self.transcription_text.setLineWrapMode(QTextEdit.WidgetWidth)
        self.transcription_text.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                background-color: white;
            }}
        """)
        transcription_layout.addWidget(self.transcription_text)

        # Add transcription tab
        self.tabs.addTab(transcription_tab, "🎙️ Transcription")

        # Create tab for generated content
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)

        # Output header
        output_header = QHBoxLayout()
        output_label = QLabel("Generated Content")
        output_label.setFont(QFont("Arial", 14, QFont.Bold))
        output_label.setStyleSheet(f"color: {COLORS['primary']};")

        # Content type selector with styling
        self.content_type_selector = QComboBox()
        self.content_type_selector.addItems(["Summary", "Meeting Minutes", "Action Items"])
        self.content_type_selector.setMinimumHeight(36)
        self.content_type_selector.setStyleSheet(f"""
            QComboBox {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px 10px;
                background-color: white;
                color: {COLORS['dark']};
                min-width: 150px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 20px;
                border-left: 1px solid #ddd;
            }}
        """)

        # Generate content button
        self.generate_button = StyledButton("Generate")
        self.generate_button.clicked.connect(self.generate_content)

        output_header.addWidget(output_label)
        output_header.addStretch()
        output_header.addWidget(QLabel("Type:"))
        output_header.addWidget(self.content_type_selector)
        output_header.addWidget(self.generate_button)

        output_layout.addLayout(output_header)

        # Output text area with styled border
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Arial", 11))
        self.output_text.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                background-color: white;
            }}
        """)
        output_layout.addWidget(self.output_text)

        # Add output tab with icon
        self.tabs.addTab(output_tab, "📝 Generated Content")

        # Create and add the Chat with AI tab
        self.chat_tab = ChatWithAITab()
        self.tabs.addTab(self.chat_tab, "💬 Chat with AI")

        # Set the transcription tab (index 0) as the default tab
        self.tabs.setCurrentIndex(0)

        # Control panel with background
        control_panel_container = QFrame()
        control_panel_container.setStyleSheet(f"""
            QFrame {{
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #ddd;
            }}
        """)
        control_panel = QHBoxLayout(control_panel_container)
        control_panel.setContentsMargins(20, 15, 20, 15)

        # Agent control
        self.agent_toggle_button = StyledButton("Start Agent")
        self.agent_toggle_button.setCheckable(True)
        self.agent_toggle_button.clicked.connect(self.toggle_agent)
        self.agent_toggle_button.setMinimumWidth(140)

        # Status indicator with styled background
        self.status_container = QFrame()
        self.status_container.setStyleSheet(f"""
            QFrame {{
                background-color: #f0f0f0;
                border-radius: 5px;
                padding: 2px;
            }}
        """)
        status_layout = QHBoxLayout(self.status_container)
        status_layout.setContentsMargins(10, 5, 10, 5)

        self.status_indicator = QLabel("Status: Inactive")
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("font-weight: bold;")
        self.status_indicator.setMinimumWidth(150)
        status_layout.addWidget(self.status_indicator)

        # Document upload button
        self.upload_button = StyledButton("Upload Document", primary=False)
        self.upload_button.clicked.connect(self.upload_document)
        self.upload_button.setMinimumWidth(170)
        self.upload_button.setIcon(QIcon.fromTheme("document-open"))

        # Dark mode toggle
        self.dark_mode_checkbox = QCheckBox("Dark Mode")
        self.dark_mode_checkbox.setStyleSheet("""
            QCheckBox {
                spacing: 8px;
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #ccc;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border: 2px solid #3498db;
            }
        """)
        self.dark_mode_checkbox.stateChanged.connect(self.toggle_dark_mode)

        control_panel.addWidget(self.agent_toggle_button)
        control_panel.addWidget(self.status_container)
        control_panel.addStretch()
        control_panel.addWidget(self.upload_button)
        control_panel.addWidget(self.dark_mode_checkbox)

        content_layout.addWidget(control_panel_container)

        # Status bar with styling
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
                padding: 5px;
            }}
        """)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(f"Meeting Copilot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def setup_file_monitoring(self):
        """Set up timers to monitor transcription and output files"""
        self.transcription_timer = QTimer(self)
        self.transcription_timer.timeout.connect(self.check_transcription_updates)
        self.transcription_timer.start(1000)  # Check every second

        self.output_timer = QTimer(self)
        self.output_timer.timeout.connect(self.check_output_updates)
        self.output_timer.start(1000)  # Check every second

    def check_transcription_updates(self):
        """Check for updates to the transcription file"""
        if not TRANSCRIPTION_FILE.exists():
            return

        current_size = TRANSCRIPTION_FILE.stat().st_size
        if current_size != self.transcription_last_size:
            try:
                with open(TRANSCRIPTION_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.signals.transcription_updated.emit(content)
                self.transcription_last_size = current_size
            except Exception as e:
                print(f"Error reading transcription file: {e}")

    def check_output_updates(self):
        """Check for updates to the output file"""
        if not OUTPUT_FILE.exists():
            return

        current_size = OUTPUT_FILE.stat().st_size
        if current_size != self.output_last_size:
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content != self.last_output_content:
                        self.last_output_content = content
                        self.signals.output_updated.emit(content)

                        # Switch to output tab when new output is generated
                        self.tabs.setCurrentIndex(1)  # Index 1 is the output tab
                self.output_last_size = current_size
            except Exception as e:
                print(f"Error reading output file: {e}")

    def update_transcription(self, text):
        """Update the transcription text area with styled text"""
        # Format the text to enhance readability
        formatted_text = text

        # Apply different colors to different speakers (if detected)
        speaker_pattern = r'(Speaker \d+:|Person \d+:)'
        if any(speaker in text for speaker in ['Speaker', 'Person']):
            speakers = set()
            for line in text.split('\n'):
                for match in ['Speaker', 'Person']:
                    if match in line:
                        speakers.add(line.split(':')[0])

            # Assign colors to speakers
            speaker_colors = [
                '#3498db',  # Blue
                '#e74c3c',  # Red
                '#2ecc71',  # Green
                '#9b59b6',  # Purple
                '#f39c12',  # Orange
                '#1abc9c',  # Turquoise
            ]

            # Replace speakers with colored versions
            for i, speaker in enumerate(speakers):
                color = speaker_colors[i % len(speaker_colors)]
                formatted_text = formatted_text.replace(
                    f"{speaker}:",
                    f"<span style='color:{color}; font-weight:bold;'>{speaker}:</span>"
                )

        # Set formatted text
        self.transcription_text.setHtml(formatted_text)

        # Scroll to bottom
        cursor = self.transcription_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.transcription_text.setTextCursor(cursor)

    def update_output(self, text):
        """Update the output text area with styled text"""
        # Apply formatting to headings, lists, etc.
        formatted_text = text

        # Format headings
        for h_level in range(1, 4):
            heading_chars = '#' * h_level
            formatted_text = formatted_text.replace(
                f"{heading_chars} ",
                f"<h{h_level} style='color:{COLORS['primary']};'>"
            ).replace("\n", f"</h{h_level}>\n", 1)

        # Format important phrases
        keywords = ["Action Items", "Summary", "Minutes", "Next Steps", "Participants"]
        for keyword in keywords:
            formatted_text = formatted_text.replace(
                f"{keyword}:",
                f"<span style='color:{COLORS['accent']}; font-weight:bold;'>{keyword}:</span>"
            )

        # Set text and scroll to bottom
        self.output_text.setText(formatted_text)
        cursor = self.output_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.output_text.setTextCursor(cursor)

    def toggle_agent(self):
        """Toggle the agent state with animation effects"""
        self.agent_running = not self.agent_running

        if self.agent_running:
            self.agent_toggle_button.setText("Stop Agent")
            self.agent_toggle_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['accent']};
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #c0392b;
                }}
                QPushButton:pressed {{
                    background-color: #c0392b;
                    padding: 9px 15px 7px 17px;
                }}
            """)
            self.signals.agent_status_changed.emit("Active - Listening for Wake Word")
            self.statusBar.showMessage("Agent started. Say 'Alexa' to activate.")

            # Animate status change
            self.animate_status_change("Active")
        else:
            self.agent_toggle_button.setText("Start Agent")
            self.agent_toggle_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['primary']};
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #2980b9;
                }}
                QPushButton:pressed {{
                    background-color: #2980b9;
                    padding: 9px 15px 7px 17px;
                }}
            """)
            self.signals.agent_status_changed.emit("Inactive")
            self.statusBar.showMessage("Agent stopped.")

            # Animate status change
            self.animate_status_change("Inactive")

    def animate_status_change(self, status):
        """Animate the status change with a visual effect"""
        # Create a flash effect on the status indicator
        original_style = self.status_container.styleSheet()

        # Flash to highlight color
        if status == "Active":
            flash_color = COLORS["success"]
        elif status == "Processing":
            flash_color = COLORS["warning"]
        else:
            flash_color = "#f0f0f0"

        self.status_container.setStyleSheet(f"""
            QFrame {{
                background-color: {flash_color};
                border-radius: 5px;
                padding: 2px;
            }}
        """)

        # Return to normal after a delay
        QTimer.singleShot(300, lambda: self.status_container.setStyleSheet(original_style))

    def update_agent_status(self, status):
        """Update the status indicator with animation"""
        self.status_indicator.setText(f"Status: {status}")

        # Update status indicator appearance based on status
        if "Active" in status:
            self.status_container.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['success']};
                    border-radius: 5px;
                    padding: 2px;
                }}
            """)
            self.status_indicator.setStyleSheet("color: white; font-weight: bold;")
        elif "Processing" in status:
            self.status_container.setStyleSheet(f"""
                QFrame {{
                    background-color: {COLORS['warning']};
                    border-radius: 5px;
                    padding: 2px;
                }}
            """)
            self.status_indicator.setStyleSheet("color: white; font-weight: bold;")
        else:
            self.status_container.setStyleSheet("""
                QFrame {
                    background-color: #f0f0f0;
                    border-radius: 5px;
                    padding: 2px;
                }
            """)
            self.status_indicator.setStyleSheet("color: #555; font-weight: bold;")

    def upload_document(self):
        """Handle document upload for vector database with improved UI feedback"""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Upload Documents",
            "",
            "All Files (*);;PDF Files (*.pdf);;Text Files (*.txt);;Word Documents (*.docx)",
            options=options
        )

        if file_paths:
            try:
                # Create the storage directory if it doesn't exist
                storage_dir = Path("storage")
                if not storage_dir.exists():
                    storage_dir.mkdir(parents=True)

                # Show processing animation by temporarily changing status
                self.signals.agent_status_changed.emit("Processing")
                self.statusBar.showMessage("Processing documents...")

                # Copy files to storage directory
                for file_path in file_paths:
                    file_name = os.path.basename(file_path)
                    dest_path = storage_dir / file_name

                    # In a real implementation, you'd process this file with your RAG system
                    # For now, we'll just copy the file to simulate the behavior
                    import shutil
                    shutil.copy2(file_path, dest_path)

                # Restore status after a delay to simulate processing
                QTimer.singleShot(1000, lambda:
                self.signals.agent_status_changed.emit("Active - Listening for Wake Word"
                                                       if self.agent_running else "Inactive"))

                # Show success message with more details
                QMessageBox.information(
                    self,
                    "Upload Successful",
                    f"{len(file_paths)} document(s) uploaded successfully.\n\n"
                    f"Files: {', '.join([os.path.basename(path) for path in file_paths])}\n\n"
                    f"These documents will be available for reference during the meeting."
                )

                self.statusBar.showMessage(f"{len(file_paths)} document(s) uploaded successfully")

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Upload Error",
                    f"Failed to upload document(s):\n{str(e)}"
                )
                self.signals.agent_status_changed.emit("Active - Listening for Wake Word"
                                                       if self.agent_running else "Inactive")

    def generate_content(self):
        """Generate meeting content with better visual feedback"""
        content_type = self.content_type_selector.currentText().lower()

        # Disable the generate button and show processing status
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating...")
        self.signals.agent_status_changed.emit("Processing")
        self.statusBar.showMessage(f"Generating {content_type}...")

        # Create a text query based on the selected content type
        if "summary" in content_type.lower():
            query = "Generate a concise summary of this meeting highlighting the key points discussed."
        elif "minutes" in content_type.lower():
            query = "Generate detailed meeting minutes with all decisions, discussions, and action items."
        elif "action" in content_type.lower():
            query = "Extract and list all action items from this meeting with assignees if mentioned."

        # Get the current transcription
        if TRANSCRIPTION_FILE.exists():
            try:
                with open(TRANSCRIPTION_FILE, 'r', encoding='utf-8') as f:
                    transcription_content = f.read()
            except Exception as e:
                print(f"Error reading transcription file: {e}")
                transcription_content = ""
        else:
            transcription_content = ""

        def process_content_generation():
            """Process content generation in a separate thread"""
            try:
                # For demo purposes, show a simulated delay to represent processing
                time.sleep(2)

                # Fallback to demo content generation
                self._generate_fallback_content(content_type, transcription_content)

                # Re-enable the generate button and restore status
                self.generate_button.setEnabled(True)
                self.generate_button.setText("Generate")
                self.signals.agent_status_changed.emit(
                    "Active - Listening for Wake Word" if self.agent_running else "Inactive")
                self.statusBar.showMessage(f"{content_type.capitalize()} generated successfully!")

            except Exception as e:
                print(f"Error in content generation: {e}")
                # Re-enable the generate button and restore status
                self.generate_button.setEnabled(True)
                self.generate_button.setText("Generate")
                self.signals.agent_status_changed.emit(
                    "Active - Listening for Wake Word" if self.agent_running else "Inactive")
                self.statusBar.showMessage(f"Error generating {content_type}: {str(e)}")

        # Run in a separate thread
        thread = threading.Thread(target=process_content_generation)
        thread.daemon = True
        thread.start()

    def _generate_fallback_content(self, content_type, transcription):
        """Generate fallback content based on transcription with improved formatting"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create basic content based on transcription
        lines = transcription.split('\n')
        words = transcription.split()

        # Set a list of participants for demo purposes
        participants = ["Alex Chen", "Sarah Johnson", "Michael Lee", "Emma Davis"]

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"# {content_type.upper()} GENERATED AT {current_time}\n\n")

            if "summary" in content_type.lower():
                text = ""
                client = genai.Client(
                    api_key=API_KEY,
                )

                model = CONTENT_MODEL

                with open("meeting_storage/transcription.txt", 'r', encoding='utf-8') as file:
                    raw_content = file.read()
                    # Filter out lines starting with "Alexa" and ending with "mute"
                    import re
                    content = re.sub(r'alexa.*?mute', '', raw_content, flags=re.IGNORECASE)
                    print(f"Filtered content: {content}")
                compiled_query = f"{content}\n\nUser Request: Summarize the meeting."
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

                print(text)
                f.write(text)


            elif "minutes" in content_type.lower():
                text = ""
                client = genai.Client(
                    api_key=API_KEY,
                )

                model = CONTENT_MODEL

                with open("meeting_storage/transcription.txt", 'r', encoding='utf-8') as file:
                    raw_content = file.read()
                    # Filter out lines starting with "Alexa" and ending with "mute"
                    import re
                    content = re.sub(r'alexa.*?mute', '', raw_content, flags=re.IGNORECASE)
                    print(f"Filtered content: {content}")
                compiled_query = f"{content}\n\nUser Request: Generate the minutes of the meeting."
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

                f.write(text)


            elif "action" in content_type.lower():
                text = ""
                client = genai.Client(
                    api_key=API_KEY,
                )

                model = CONTENT_MODEL

                with open("meeting_storage/transcription.txt", 'r', encoding='utf-8') as file:
                    raw_content = file.read()
                    # Filter out lines starting with "Alexa" and ending with "mute"
                    import re
                    content = re.sub(r'alexa.*?mute', '', raw_content, flags=re.IGNORECASE)
                    print(f"Filtered content: {content}")
                compiled_query = f"{content}\n\nUser Request: Generate the action item of the meeting."
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
                        types.Part.from_text(
                            text="""You are an AI assistant specialized in analyzing and summarizing meeting transcriptions with precision and accuracy. Your task is to extract meaningful information from the provided meeting transcription while adhering to the following detailed guidelines. While giving the answer no need to mention when that transcription was generated. Only give answer to specific query."""),
                    ],
                )

                for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                ):
                    text += chunk.text + " "

                f.write(text)

    def toggle_dark_mode(self, state):
        """Toggle dark mode for the UI with smoother transitions"""
        app = QApplication.instance()
        self.is_dark_mode = (state == Qt.Checked)

        if self.is_dark_mode:
            # Dark theme
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(COLORS["dark_mode_bg"]))
            dark_palette.setColor(QPalette.WindowText, QColor(COLORS["dark_mode_text"]))
            dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ToolTipText, QColor(COLORS["dark_mode_text"]))
            dark_palette.setColor(QPalette.Text, QColor(COLORS["dark_mode_text"]))
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, QColor(COLORS["dark_mode_text"]))
            dark_palette.setColor(QPalette.BrightText, QColor(COLORS["accent"]))
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))

            app.setPalette(dark_palette)

            # Update text editors
            self.transcription_text.setStyleSheet("""
                QTextEdit {
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 10px;
                    background-color: #2c3e50;
                    color: #ecf0f1;
                }
            """)

            self.output_text.setStyleSheet("""
                QTextEdit {
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 10px;
                    background-color: #2c3e50;
                    color: #ecf0f1;
                }
            """)

            # Update status bar
            self.statusBar.setStyleSheet("""
                QStatusBar {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    font-weight: bold;
                    padding: 5px;
                }
            """)

            # Update tabs
            self.tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 5px;
                    background-color: #1e272e;
                }

                QTabBar::tab {
                    background-color: #2c3e50;
                    color: #bdc3c7;
                    border: 1px solid #444;
                    border-bottom: none;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    padding: 8px 16px;
                    min-width: 150px;
                    font-weight: normal;
                }

                QTabBar::tab:selected {
                    background-color: #1e272e;
                    color: #3498db;
                    font-weight: bold;
                    border-bottom: 2px solid #3498db;
                }

                QTabBar::tab:hover:!selected {
                    background-color: #34495e;
                }
            """)

            # Update control panel
            control_panel_container = self.findChild(QFrame)
            if control_panel_container:
                control_panel_container.setStyleSheet("""
                    QFrame {
                        background-color: #2c3e50;
                        border-radius: 8px;
                        border: 1px solid #444;
                    }
                """)

        else:
            # Light theme
            app.setPalette(app.style().standardPalette())

            # Update text editors
            self.transcription_text.setStyleSheet("""
                QTextEdit {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 10px;
                    background-color: white;
                }
            """)

            self.output_text.setStyleSheet("""
                QTextEdit {
                                border: 1px solid #ddd;
                                border-radius: 8px;
                                padding: 10px;
                                background-color: white;
                            }
                        """)

            # Update status bar
            self.statusBar.setStyleSheet(f"""
                            QStatusBar {{
                                background-color: {COLORS['primary']};
                                color: white;
                                font-weight: bold;
                                padding: 5px;
                            }}
                        """)

            # Update tabs
            self.tabs.setStyleSheet("""
                            QTabWidget::pane {
                                border: 1px solid #ddd;
                                border-radius: 4px;
                                padding: 5px;
                                background-color: white;
                            }

                            QTabBar::tab {
                                background-color: #f0f0f0;
                                color: #555;
                                border: 1px solid #ddd;
                                border-bottom: none;
                                border-top-left-radius: 4px;
                                border-top-right-radius: 4px;
                                padding: 8px 16px;
                                min-width: 150px;
                                font-weight: normal;
                            }

                            QTabBar::tab:selected {
                                background-color: white;
                                color: #3498db;
                                font-weight: bold;
                                border-bottom: 2px solid #3498db;
                            }

                            QTabBar::tab:hover:!selected {
                                background-color: #e0e0e0;
                            }
                        """)

            # Update control panel
            control_panel_container = self.findChild(QFrame)
            if control_panel_container:
                control_panel_container.setStyleSheet("""
                                QFrame {
                                    background-color: #f8f9fa;
                                    border-radius: 8px;
                                    border: 1px solid #ddd;
                                }
                            """)

        def main():
            app = QApplication(sys.argv)
            app.setStyle("Fusion")  # Use Fusion style for a clean, modern look

            # Set application font
            font = QFont("Arial", 10)
            app.setFont(font)

            window = MeetingCopilotUI()
            window.show()

            sys.exit(app.exec_())

        if __name__ == "__main__":
            main()