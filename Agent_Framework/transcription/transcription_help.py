import speechmatics
import speechmatics.models
import speechmatics.client
import speechmatics.cli
import asyncio
import argparse
import sys
import soundcard as sc
import numpy as np
import datetime
import os
import traceback
import time


import json
import logging
import os
import ssl
import sys
from dataclasses import dataclass
from socket import gaierror
from typing import Any, Dict, List

import httpx
import toml
from websockets.exceptions import WebSocketException

import speechmatics.adapters
from speechmatics.batch_client import BatchClient
from speechmatics.cli_parser import parse_args
from speechmatics.client import WebsocketClient
from speechmatics.config import read_config_from_home
from speechmatics.constants import BATCH_SELF_SERVICE_URL, RT_SELF_SERVICE_URL
from speechmatics.exceptions import JobNotFoundException, TranscriptionError
from speechmatics.helpers import _process_status_errors
from ..audio.audio_utils import AudioUtils
from speechmatics.models import (
    AudioEventsConfig,
    AudioSettings,
    AutoChaptersConfig,
    BatchLanguageIdentificationConfig,
    BatchSpeakerDiarizationConfig,
    BatchTranscriptionConfig,
    ClientMessageType,
    ConnectionSettings,
    RTSpeakerDiarizationConfig,
    RTTranslationConfig,
    ServerMessageType,
    SentimentAnalysisConfig,
    SummarizationConfig,
    TopicDetectionConfig,
    TranscriptionConfig,
)

from pathlib import Path

from ..config.settings import PERSIST_DIR, gemini_embedding_model, llm

import os
from llama_index.core import Document, StorageContext, load_index_from_storage, VectorStoreIndex, Settings

# pylint: disable=too-many-arguments,too-many-statements
def add_printing_handlers(
    api,
    transcripts,
    word_list,
    live_buffer,
    live_doc_ids,
    enable_partials=False,
    enable_transcription_partials=False,
    enable_translation_partials=False,
    debug_handlers_too=False,
    print_json=False,
    translation_config=None,
):
    """
    Adds a set of handlers to the websocket client which print out transcripts
    as they are received. This includes partials if they are enabled.

    Args:
        api (speechmatics.client.WebsocketClient): Client instance.
        transcripts (Transcripts): Allows the transcripts to be concatenated to
            produce a final result.
        enable_partials (bool, optional): Whether partials are enabled
            for both transcription and translation.
        enable_transcription_partials (bool, optional): Whether partials are enabled
            for transcription only.
        enable_translation_partials (bool, optional): Whether partials are enabled
            for translation only.
        debug_handlers_too (bool, optional): Whether to enable 'debug'
            handlers that print out an ASCII symbol representing messages being
            received and sent.
        print_json (bool, optional): Whether to print json transcript messages.
        translation_config (TranslationConfig, optional): Translation config with target languages.
    """
    escape_seq = "\33[2K" if sys.stdout.isatty() else ""

    if debug_handlers_too:
        api.add_event_handler(
            ServerMessageType.AudioAdded, lambda *args: print_symbol("-")
        )
        api.add_event_handler(
            ServerMessageType.AddPartialTranscript, lambda *args: print_symbol(".")
        )
        api.add_event_handler(
            ServerMessageType.AddTranscript, lambda *args: print_symbol("|")
        )
        api.add_middleware(ClientMessageType.AddAudio, lambda *args: print_symbol("+"))

    def partial_transcript_handler(message):
        # "\n" does not appear in partial transcripts
        if print_json:
            print(json.dumps(message))
            return
        plaintext = speechmatics.adapters.convert_to_txt(
            message["results"],
            api.transcription_config.language,
            language_pack_info=api.get_language_pack_info(),
            speaker_labels=True,
        )
        if plaintext:
            sys.stderr.write(f"{escape_seq}{plaintext}\r")

    def insert_document(index, doc):
        """Insert and return doc_id if your vectorstore supports it."""
        return index.insert(doc)

    def transcript_handler(message):
        nonlocal word_list
        nonlocal live_buffer
        nonlocal live_doc_ids
        transcripts.json.append(message)
        if print_json:
            print(json.dumps(message))
            return
        plaintext = speechmatics.adapters.convert_to_txt(
            message["results"],
            api.transcription_config.language,
            language_pack_info=api.get_language_pack_info(),
            speaker_labels=True,
        )
        if plaintext:
            word_list.append(plaintext)
            live_buffer.append(plaintext)

            # Insert live chunk every 20 words
            if len(live_buffer) >= 20:
                live_text = ' '.join(live_buffer)
                live_doc = Document(
                    text=live_text,
                    metadata={"type": "live", "timestamp": time.time()}
                )

                # Initialize index if needed
                Settings.llm = llm
                Settings.embed_model = gemini_embedding_model

                if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
                    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                    index = load_index_from_storage(storage_context)
                else:
                    index = VectorStoreIndex.from_documents([live_doc])
                    storage_context = index.storage_context  # capture this

                # Insert and track live doc ID
                doc_id = insert_document(index, live_doc)
                live_doc_ids.append(doc_id)

                index.storage_context.persist(persist_dir=PERSIST_DIR)

                # Retain last few words in buffer for overlap
                live_buffer = live_buffer[-5:]

            # Insert stable chunk every 200 words
            if len(word_list) >= 200:
                full_text = ' '.join(word_list[:200])
                stable_doc = Document(
                    text=full_text,
                    metadata={"type": "stable", "timestamp": time.time()}
                )

                Settings.llm = llm
                Settings.embed_model = gemini_embedding_model

                if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
                    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                    index = load_index_from_storage(storage_context)

                insert_document(index, stable_doc)

                # Remove previously inserted live docs
                for doc_id in live_doc_ids:
                    index.delete(doc_id)
                live_doc_ids = []

                index.storage_context.persist(persist_dir=PERSIST_DIR)

                # Retain last 50 words for overlap
                word_list = word_list[-50:]

            sys.stdout.write(str(word_list)+"\n")
            sys.stdout.write(f"{escape_seq}{plaintext}\n")

            with open(transcripts.output_file, 'a', encoding='utf-8') as f:
                f.write(f"{plaintext} ")
        transcripts.text += plaintext

    def audio_event_handler(message):
        if print_json:
            print(json.dumps(message))
            return
        event_name = message["event"].get("type", "").upper()
        sys.stdout.write(f"{escape_seq}[{event_name}]\n")
        transcripts.text += f"[{event_name}] "

    def partial_translation_handler(message):
        if print_json:
            print(json.dumps(message))
            return

        if translation_config.target_languages[0] == message["language"]:
            plaintext = speechmatics.adapters.get_txt_translation(message["results"])
            sys.stderr.write(f"{escape_seq}{plaintext}\r")

    def translation_handler(message):
        transcripts.json.append(message)
        if print_json:
            print(json.dumps(message))
            return

        if translation_config.target_languages[0] == message["language"]:
            plaintext = speechmatics.adapters.get_txt_translation(message["results"])
            if plaintext:
                sys.stdout.write(f"{escape_seq}{plaintext}\n")
            transcripts.text += plaintext

    def end_of_transcript_handler(_):
        if enable_partials:
            print("\n", file=sys.stderr)

    api.add_event_handler(ServerMessageType.EndOfTranscript, end_of_transcript_handler)

    api.add_event_handler(ServerMessageType.AudioEventStarted, audio_event_handler)
    if print_json:
        if enable_partials or enable_translation_partials:
            api.add_event_handler(
                ServerMessageType.AddPartialTranslation,
                partial_translation_handler,
            )
        api.add_event_handler(ServerMessageType.AddTranslation, translation_handler)
        if enable_partials or enable_transcription_partials:
            api.add_event_handler(
                ServerMessageType.AddPartialTranscript,
                partial_transcript_handler,
            )
        api.add_event_handler(ServerMessageType.AddTranscript, transcript_handler)
    else:
        if translation_config is not None:
            if enable_partials or enable_translation_partials:
                api.add_event_handler(
                    ServerMessageType.AddPartialTranslation,
                    partial_translation_handler,
                )
            api.add_event_handler(ServerMessageType.AddTranslation, translation_handler)
        else:
            if enable_partials or enable_transcription_partials:
                api.add_event_handler(
                    ServerMessageType.AddPartialTranscript,
                    partial_transcript_handler,
                )
            api.add_event_handler(ServerMessageType.AddTranscript, transcript_handler)


class SoundcardStreamWrapper:
    def __init__(self, speaker, microphone, sample_rate):
        """
        Initialize a wrapper for soundcard's audio capture

        Args:
            speaker: The soundcard speaker object to capture from
            microphone: The soundcard microphone object to capture from
            sample_rate: The sample rate to use for recording
        """
        self.speaker = speaker
        self.microphone = microphone
        self.sample_rate = sample_rate
        self.speaker_recorder = None
        self.mic_recorder = None
        self.chunk_frames = 1024  # Number of frames to read at once

    def __enter__(self):
        # Create recorders for both speaker and microphone
        if self.speaker:
            self.speaker_recorder = sc.get_microphone(
                id=str(self.speaker.name),
                include_loopback=True
            ).recorder(samplerate=self.sample_rate)
            self.speaker_recorder.__enter__()

        if self.microphone:
            self.mic_recorder = sc.get_microphone(
                id=str(self.microphone.name),
                include_loopback=False  # Regular mic recording, not loopback
            ).recorder(samplerate=self.sample_rate)
            self.mic_recorder.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.speaker_recorder:
            self.speaker_recorder.__exit__(exc_type, exc_val, exc_tb)
        if self.mic_recorder:
            self.mic_recorder.__exit__(exc_type, exc_val, exc_tb)

    def read(self, frames):
        """Read audio data from both speaker loopback and microphone

        Mixes both audio sources and converts to the format expected by Speechmatics
        """
        if not self.speaker_recorder and not self.mic_recorder:
            return bytes()

        # Initialize with zeros if both recorders aren't available
        mixed_audio = np.zeros(self.chunk_frames, dtype=np.float32)

        # Read and mix speaker audio if available
        if self.speaker_recorder:
            speaker_data = self.speaker_recorder.record(numframes=self.chunk_frames)
            # Take first channel if stereo
            speaker_mono = speaker_data[:, 0] if speaker_data.ndim > 1 else speaker_data
            mixed_audio += speaker_mono

        # Read and mix microphone audio if available
        if self.mic_recorder:
            mic_data = self.mic_recorder.record(numframes=self.chunk_frames)
            # Take first channel if stereo
            mic_mono = mic_data[:, 0] if mic_data.ndim > 1 else mic_data
            mixed_audio += mic_mono

        # Normalize the mixed audio (prevent clipping)
        # Only normalize if the max amplitude is greater than 1.0
        max_amp = np.max(np.abs(mixed_audio))
        if max_amp > 1.0:
            mixed_audio = mixed_audio / max_amp

        # Convert to bytes in the expected format (pcm_f32le/be)
        return mixed_audio.tobytes()

class FileWriterTranscripts(speechmatics.cli.Transcripts):
    """Extension of Speechmatics Transcripts that writes to a file in real-time"""

    def __init__(self, text="", json=None, output_file=None):
        super().__init__(text=text, json=json or [])
        self.output_file = output_file
        self.last_transcript = ""
        self.transcript_count = 0

        # Create/open the file and write a header with timestamp
        if self.output_file:
            try:
                # Make sure the directory exists
                output_dir = os.path.dirname(self.output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    f.write(f"===== Transcription started at {timestamp} =====\n\n")
                print(f"Successfully created file: {os.path.abspath(self.output_file)}")
            except Exception as e:
                print(f"Error creating file: {e}")
                traceback.print_exc()
                raise

    def __call__(self, transcript):
        """Called when a new transcript is received"""
        self.on_transcript(transcript)  # Ensure on_transcript is called

    def on_transcript(self, transcript):
        """Called when a new transcript is received"""
        # Call the parent method to update internal state
        super().on_transcript(transcript)

        # Write the new transcript to the file
        if self.output_file and transcript != self.last_transcript:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    # For a complete transcript (not partial), add a timestamp
                    if not transcript.get("is_partial", True):
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        f.write(f"[{timestamp}] {transcript['content']}\n")
                        print(f"Saved transcript #{self.transcript_count} to file")
                        self.transcript_count += 1

                self.last_transcript = transcript
            except Exception as e:
                print(f"Error writing to file: {e}")
                traceback.print_exc()


async def transcribe_from_speaker(speechmatics_client, language: str, max_delay):
    frame_rate = 16_000  # Speechmatics works better with 16kHz

    # List available speakers
    mics, speakers = AudioUtils.list_audio_devices()
    print(speakers)
    microphone = mics[0]
    speaker = speakers[1]
    if not speakers:
        print("No speakers found!")
        return

    # Use the first speaker
    print(f"Using speaker: {speaker.name}")

    with SoundcardStreamWrapper(speaker, microphone, frame_rate) as stream:
        settings = speechmatics.models.AudioSettings(
            sample_rate=frame_rate,
            encoding="pcm_f32" + ("le" if sys.byteorder == "little" else "be"),
        )

        conf = speechmatics.models.TranscriptionConfig(
            language=language,
            operating_point="enhanced",
            max_delay=1,
            enable_partials=True,
            enable_entities=True,
        )
        await speechmatics_client.run(stream, conf, settings)
