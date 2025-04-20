"""
Wake word detection functionality
Current Date and Time (UTC): 2025-04-16 17:36:12
Current User's Login: SAMBITMALLICK2003
"""

import threading
import time
import traceback
import numpy as np
import warnings
from openwakeword.model import Model
from ..audio.audio_utils import AudioUtils
from ..config.settings import WAKEWORD_CHUNK_SIZE, WAKEWORD_THRESHOLD, SEND_SAMPLE_RATE
import soundcard as sc
from pathlib import Path

# Filter out the specific soundcard warning
from soundcard.mediafoundation import SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning,
                        message="data discontinuity in recording")

class WakeWordDetector:
    """Detects wake words using OpenWakeWord"""

    def __init__(self, model_path, inference_framework='tflite'):
        self.model_path = model_path
        self.inference_framework = inference_framework
        self.chunk_size = WAKEWORD_CHUNK_SIZE
        self.running = False
        self.detected = False
        self.detector_thread = None
        self.owwModel = None


    def start(self):
        self.running = True
        self.detected = False
        self.detector_thread = threading.Thread(target=self._run_detection)
        self.detector_thread.daemon = True
        self.detector_thread.start()

    def stop(self):
        self.running = False
        if self.detector_thread:
            self.detector_thread.join(timeout=1)

    def reset(self):
        self.detected = False

    def _run_detection(self):
        # Initialize wake word model
        self.owwModel = Model(
            wakeword_models=[self.model_path],
            inference_framework=self.inference_framework
        )

        # Find default speaker to capture from without printing devices
        try:
            mics, speakers= AudioUtils.list_audio_devices()
            speaker = speakers[2]
            microphone = mics[1]
            if not speaker:
                print("Error: No speakers found for audio capture")
                return
            if not microphone:
                print("Error: No microphone found for audio capture")
                return
        except Exception as e:
            print(f"Error finding default speaker: {e}")
            return

        print("\nListening for the wake word...")

        # Use the loopback recorder
        try:
            with (sc.get_microphone(id=str(microphone.name), include_loopback=True).recorder(
                    samplerate=SEND_SAMPLE_RATE) as mic):
                while self.running:
                    # Record a chunk of audio
                    audio_float = mic.record(numframes=self.chunk_size)

                    # Convert from float32 [-1.0, 1.0] to int16 for compatibility with OWW
                    audio_mono = audio_float[:, 0]  # Take first channel if stereo
                    audio_data = (audio_mono * 32767).astype(np.int16)

                    # Feed to openWakeWord model
                    prediction = self.owwModel.predict(audio_data)

                    # Get the score for the single model
                    model_name = list(self.owwModel.models.keys())[0]
                    score = 0
                    # score = self.owwModel.prediction_buffer[model_name][-1]


                    file_path = str(Path(__file__).resolve().parent.parent.parent / "meeting_storage" / "transcription.txt")
                    # Read file content
                    if not self.detected:
                        with open(file_path, "r", encoding="utf-8") as file:
                            content = file.read().strip()  # Strip to remove trailing newlines/spaces

                        # Extract words and get the last 4
                        words = content.split(" ")
                        last_4_words = words[-4:]

                        # Check if the last 5 letters are 'alexa'
                        is_elsa = False
                        for word in last_4_words[::-1]:
                            if "mute" in word.lower():
                                break
                            if "alexa" in word.lower():
                                is_elsa = True
                                break

                        if is_elsa:
                            score = 1
                            is_elsa = False


                        # Print the result
                        if score > WAKEWORD_THRESHOLD and not self.detected:
                            print(f"\nWake word Detected! (Score: {score:.2f})")
                            self.detected = True
                            with open(str(Path(__file__).resolve().parent.parent.parent / "output_storage" / "var.txt"),
                                      "w") as file:
                                file.write(str(self.detected))
                        elif not self.detected:
                            # Less frequent updates for smoother display
                            if round(time.time() * 2) % 2 == 0:  # Update roughly twice per second
                                print(f"Listening for wake word... (Score: {score:.2f})", end="\r")
        except Exception as e:
            print(f"Wake word detection error: {e}")
            traceback.print_exc()

        print("Wake word detector stopped")


class SleepWordDetector(WakeWordDetector):
    """Detects sleep/stop words using OpenWakeWord"""

    def _run_detection(self):
        # Initialize wake word model
        self.owwModel = Model(
            wakeword_models=[self.model_path],
            inference_framework=self.inference_framework
        )

        # Find default speaker to capture from without printing devices
        try:
            mics, speakers = AudioUtils.list_audio_devices()
            speaker = speakers[2]
            microphone = mics[1]
            if not speaker:
                print("Error: No speakers found for audio capture")
                return
            if not microphone:
                print("Error: No speakers found for audio capture")
                return
        except Exception as e:
            print(f"Error finding default speaker: {e}")
            return

        print("\nSleep word detector initialized...")

        # Use the loopback recorder
        try:
            with sc.get_microphone(id=str(microphone.name), include_loopback=True).recorder(
                    samplerate=SEND_SAMPLE_RATE) as mic:
                while self.running:
                    # Record a chunk of audio
                    audio_float = mic.record(numframes=self.chunk_size)

                    # Convert from float32 [-1.0, 1.0] to int16 for compatibility with OWW
                    audio_mono = audio_float[:, 0]  # Take first channel if stereo
                    audio_data = (audio_mono * 32767).astype(np.int16)

                    # Feed to openWakeWord model
                    prediction = self.owwModel.predict(audio_data)

                    # Get the score for the single model
                    model_name = list(self.owwModel.models.keys())[0]
                    score = 0
                    # score = self.owwModel.prediction_buffer[model_name][-1]

                    file_path = str(
                        Path(__file__).resolve().parent.parent.parent / "meeting_storage" / "transcription.txt")
                    # Read file content
                    if not self.detected:
                        with open(file_path, "r", encoding="utf-8") as file:
                            content = file.read().strip()  # Strip to remove trailing newlines/spaces

                            # Extract words and get the last 4
                            words = content.split(" ")
                            last_4_words = words[-4:]

                            # Check if the last 5 letters are 'alexa'
                            is_mute = False
                            for word in last_4_words[::-1]:
                                if "alexa" in word.lower():
                                    break
                                if "mute" in word.lower():
                                    is_mute = True
                                    break

                            if is_mute:
                                score = 1
                                is_mute = False

                        # Print the result with less verbose output
                        if score > WAKEWORD_THRESHOLD and not self.detected:
                            print(f"\nSleep word Detected! (Score: {score:.2f})")
                            self.detected = True
                            with open(str(Path(__file__).resolve().parent.parent.parent / "output_storage" / "var.txt"),
                                      "w") as file:
                                file.write(str(not self.detected))
                        # Removed frequent updates for cleaner terminal output
        except Exception as e:
            print(f"Sleep word detection error: {e}")
            traceback.print_exc()

        print("Sleep word detector stopped")