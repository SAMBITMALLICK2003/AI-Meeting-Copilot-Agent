import asyncio
import time
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, 
    QSplitter, QFrame, QScrollArea, QSizePolicy, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QSize
from PyQt5.QtGui import QIcon, QFont, QColor, QTextCursor, QPixmap
from google import genai
from google.genai import types
from Agent_Framework.config.settings import API_KEY, CONTENT_MODEL, llm, gemini_embedding_model
from llama_index.core import StorageContext, load_index_from_storage

# Constants
PERSIST_DIR = "./storage"
TRANSCRIPTION_FILE = Path("meeting_storage/transcription.txt")
FACT_TRIGGER_THRESHOLD = 0.75  # Threshold for similarity matching

# Phrases that indicate numerical facts or important information
FACT_TRIGGER_PHRASES = [
    "the percentage is",
    "the number of",
    "according to the data",
    "statistics show",
    "the total amount",
    "the report shows",
    "the growth rate",
    "the average",
    "the median",
    "the market share",
    "the revenue is",
    "the cost is",
    "the profit margin",
    "the deadline is",
    "the timeline for",
    "the budget is",
    "the target is",
    "the goal is",
    "the kpi is",
    "in conclusion",
    "key takeaway",
    "important finding",
    "critical information",
    "significant result"
]

class AsyncChatSignals(QObject):
    """Class to emit signals from async functions to the Qt main thread"""
    message_received = pyqtSignal(str, str)  # sender, message
    fact_detected = pyqtSignal(str, str)  # trigger phrase, relevant info
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()

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
            client = genai.Client(api_key=API_KEY)
            
            # First, try to find relevant information from RAG
            rag_response = self.query_rag(message)
            
            # Create system instruction
            system_instruction = """You are an AI assistant specialized in helping with meetings.
            You're friendly, helpful, and concise. When responding:
            1. If relevant information from documents is provided, use it to enhance your answer
            2. Be specific and detailed when answering questions
            3. If you don't know the answer, say so honestly
            4. Format your responses with markdown for better readability
            5. Keep responses concise but informative"""
            
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
            
            # Generate response
            generate_content_config = types.GenerateContentConfig(
                temperature=0.2,  # More deterministic
                response_mime_type="text/plain",
                system_instruction=[types.Part.from_text(text=system_instruction)],
            )
            
            content = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            response = client.models.generate_content(
                model=CONTENT_MODEL,
                contents=content,
                config=generate_content_config,
            )
            
            # Get response text
            response_text = response.text
            
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
            if self.last_processed_transcription and len(self.last_processed_transcription) < len(current_transcription):
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