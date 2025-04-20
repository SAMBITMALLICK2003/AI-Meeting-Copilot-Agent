"""
Utility functions for audio device management and processing
Current Date and Time (UTC): 2025-04-16 17:36:12
Current User's Login: SAMBITMALLICK2003
"""

import soundcard as sc
import warnings

# Filter out the specific warning
from soundcard.mediafoundation import SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning,
                        message="data discontinuity in recording")

class AudioUtils:
    """Utility class for audio device management and processing"""

    @staticmethod
    def list_audio_devices():
        """List all available audio devices and return them without printing"""
        # Get microphones and speakers without printing details
        mics = list(sc.all_microphones())
        speakers = list(sc.all_speakers())
        print("microphones list", mics)
        print("speakers list", speakers)
        
        return mics, speakers
        
    @staticmethod
    def get_default_speaker():
        """Get the default speaker for audio capture"""
        speakers = list(sc.all_speakers())
        print("Speakers", speakers)

        if speakers:
            return speakers[0]
        return None

    @staticmethod
    def get_default_microphone():
        """Get the default speaker for audio capture"""
        microphones = list(sc.all_microphones())
        print("Speakers", microphones)

        if microphones:
            return microphones[0]
        return None