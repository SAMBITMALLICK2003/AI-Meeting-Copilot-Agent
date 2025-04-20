"""
Voice capture functionality for the Gemini Agent Framework
Current Date and Time (UTC): 2025-04-16 17:36:12
Current User's Login: SAMBITMALLICK2003
"""

import asyncio
import wave
import numpy as np
import pyaudio
import soundcard as sc
import warnings
from ..config.settings import (
    SEND_SAMPLE_RATE, 
    SILENCE_THRESHOLD, 
    MIN_COMMAND_SILENCE_FRAMES,
    FORMAT,
    CHANNELS,
    RECEIVE_SAMPLE_RATE
)
from ..audio.audio_utils import AudioUtils

# Filter out the specific soundcard warning
from soundcard.mediafoundation import SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning,
                        message="data discontinuity in recording")

class VoiceCapture:
    """Handles voice capture and playback functionality"""
    
    def __init__(self):
        self.pya = pyaudio.PyAudio()
        
    async def listen_audio(self, out_queue, wakeword_active_fn, wakeword_detected_fn, 
                          last_activity_update_fn, speaking_check_fn, switch_to_wakeword_fn):
        """Listen for audio using soundcard for system audio capture with improved interaction"""
        # Create a WAV file to record what Gemini is hearing
        input_file = wave.open("user_input.wav", "wb")
        input_file.setnchannels(CHANNELS)
        input_file.setsampwidth(2)  # 16-bit audio (paInt16)
        input_file.setframerate(SEND_SAMPLE_RATE)
        print("\nSaving user input to user_input.wav")

        # Find default speaker for capture without printing devices
        try:
            # mics, speakers = AudioUtils.list_audio_devices()
            # speaker = speakers[0]
            # microphone = mics[1]
            speaker = AudioUtils.get_default_speaker()
            microphone = AudioUtils.get_default_microphone()
            if not speaker:
                print("Error: No speakers found for audio capture")
                if input_file:
                    input_file.close()
                return False
            if not microphone:
                print("Error: No microphone found for audio capture")
                
            print(f"Using speaker for audio capture: {speaker.name}")
        except Exception as e:
            print(f"Error finding default speaker: {e}")
            if input_file:
                input_file.close()
            return False

        buffer_after_wakeword = []
        waiting_for_command = False
        silence_count = 0

        # Use a smaller buffer size for more responsive interaction
        chunk_frames = 2048  # frames per chunk

        try:
            # Use the loopback recorder
            with sc.get_microphone(id=str(microphone.name), include_loopback=True).recorder(
                    samplerate=SEND_SAMPLE_RATE) as mic:
                print(f"Successfully started system audio capture")

                while True:
                    # Record a small chunk of audio
                    data = await asyncio.to_thread(
                        mic.record, numframes=chunk_frames
                    )

                    # Convert from float32 [-1.0, 1.0] to int16 for compatibility
                    # Take first channel if stereo
                    audio_mono = data[:, 0]
                    audio_int16 = (audio_mono * 32767).astype(np.int16)

                    # Get buffer as bytes
                    audio_bytes = audio_int16.tobytes()

                    # Save to input file (all audio is recorded to understand what Gemini is hearing)
                    input_file.writeframes(audio_bytes)

                    # Calculate audio level
                    audio_level = np.abs(audio_int16).mean()

                    # Check if wake word detection is active
                    if wakeword_active_fn():
                        # Wake word detected, prepare to capture actual command
                        if wakeword_detected_fn():
                            print("\n✓ Wake word detected, I'm listening...")
                            # Provide visual feedback
                            await switch_to_wakeword_fn(active=False)
                            waiting_for_command = True
                            buffer_after_wakeword = []
                            silence_count = 0

                            # Sending an initial greeting to make interaction more fluid
                            # await out_queue.put({"text": "I'm ready to help you."})

                    # In command listening mode (after wake word, before sending to Gemini)
                    elif waiting_for_command:
                        buffer_after_wakeword.append(audio_bytes)

                        # Check if there's actual audio content or silence
                        if audio_level > SILENCE_THRESHOLD:
                            silence_count = 0
                        else:
                            silence_count += 1

                        # If we detect a sufficient pause after speech, process the command
                        if silence_count >= MIN_COMMAND_SILENCE_FRAMES and len(buffer_after_wakeword) > 5:
                            print("Command received, processing...")
                            waiting_for_command = False
                            last_activity_update_fn()

                            # Send the buffered command audio directly
                            for audio_chunk in buffer_after_wakeword:
                                await out_queue.put({"data": audio_chunk, "mime_type": "audio/pcm"})

                            # Clear buffer
                            buffer_after_wakeword = []
                    # In conversation mode, send audio to Gemini
                    else:
                        # Don't reset inactivity timer if Gemini is currently speaking
                        if not speaking_check_fn():
                            # Check if there's actual audio content (not just silence)
                            if audio_level > SILENCE_THRESHOLD:
                                last_activity_update_fn()

                        await out_queue.put({"data": audio_bytes, "mime_type": "audio/pcm"})

        except Exception as e:
            print(f"Error in soundcard audio capture: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if input_file:
                input_file.close()

    async def play_audio(self, audio_queue):
        """Play audio responses from Gemini"""
        # Create a WAV file to record Gemini's response
        output_file = wave.open("gemini_response.wav", "wb")
        output_file.setnchannels(CHANNELS)
        output_file.setsampwidth(2)  # 16-bit audio (paInt16)
        output_file.setframerate(RECEIVE_SAMPLE_RATE)
        print("\nSaving Gemini's responses to gemini_response.wav")

        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        try:
            while True:
                bytestream = await audio_queue.get()

                # Play audio through speakers
                await asyncio.to_thread(stream.write, bytestream)

                # Also save to WAV file
                output_file.writeframes(bytestream)

        except Exception as e:
            print(f"Error in audio playback: {e}")
        finally:
            # Close the output file when the function exits
            if output_file:
                output_file.close()
            if stream:
                stream.stop_stream()
                stream.close()

    def cleanup(self):
        """Clean up resources"""
        if self.pya:
            self.pya.terminate()