"""
Text-to-Speech System
"""

import re
import platform
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Generator, Any, Protocol
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    CHINESE = "zh"
    JAPANESE = "ja"
    FRENCH = "fr"


@dataclass
class VoiceConfig:
    """Voice configuration"""
    language: Language
    voice_name: str
    speed: float = 1.2
    
    def __post_init__(self):
        if self.speed <= 0:
            raise ValueError("Speed must be positive")


class AudioPlayer(Protocol):
    """Protocol for audio playback"""
    def play(self, audio_path: Path) -> None:
        """Play audio file"""
        ...


class TextProcessor(ABC):
    """Abstract base for text processing strategies"""
    
    @abstractmethod
    def process(self, text: str) -> str:
        """Process text for TTS"""
        pass


class EnglishTextProcessor(TextProcessor):
    """English text processing"""
    
    def process(self, text: str) -> str:
        lines = text.split('\n')
        filtered = [line for line in lines if re.match(r'^\s*[a-zA-Z]', line)]
        text = ' '.join(filtered)
        
        # Remove code blocks and URLs
        text = re.sub(r'`.*?`', '', text)
        text = re.sub(r'https?://\S+', '', text)
        
        # Extract filenames from paths
        text = re.sub(r'\b[\w./\\-]+\b', self._extract_filename, text)
        text = re.sub(r'\b-\w+\b', '', text)
        text = re.sub(r'[^a-zA-Z0-9.,!? _ -]+', ' ', text)
        text = text.replace('.com', '')
        
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def _extract_filename(match: re.Match) -> str:
        path = match.group()
        parts = re.split(r'/|\\', path)
        return parts[-1] if parts else path


class ChineseTextProcessor(TextProcessor):
    """Chinese text processing"""
    
    def process(self, text: str) -> str:
        lines = text.split('\n')
        pattern = r'^\s*[\u4e00-\u9fff\uFF08\uFF3B\u300A\u3010\u201C(（\[【《]'
        filtered = [line for line in lines if re.match(pattern, line)]
        text = ' '.join(filtered)
        
        # Remove code blocks and URLs
        text = re.sub(r'`.*?`', '', text)
        text = re.sub(r'https?://\S+', '', text)
        
        # Keep only Chinese characters and punctuation
        text = re.sub(
            r'[^\u4e00-\u9fff\s，。！？《》【】""''（）()—]',
            '', 
            text
        )
        
        return re.sub(r'\s+', ' ', text).strip()


class JapaneseTextProcessor(TextProcessor):
    """Japanese text processing - placeholder for extension"""
    
    def process(self, text: str) -> str:
        # Similar to English for now, can be extended
        return EnglishTextProcessor().process(text)


class FrenchTextProcessor(TextProcessor):
    """French text processing - placeholder for extension"""
    
    def process(self, text: str) -> str:
        # Similar to English for now, can be extended
        return EnglishTextProcessor().process(text)


class TextProcessorFactory:
    """Factory for creating text processors"""
    
    _processors: Dict[Language, type[TextProcessor]] = {
        Language.ENGLISH: EnglishTextProcessor,
        Language.CHINESE: ChineseTextProcessor,
        Language.JAPANESE: JapaneseTextProcessor,
        Language.FRENCH: FrenchTextProcessor,
    }
    
    @classmethod
    def create(cls, language: Language) -> TextProcessor:
        processor_class = cls._processors.get(language)
        if not processor_class:
            raise ValueError(f"Unsupported language: {language}")
        return processor_class()


class SystemAudioPlayer:
    """System-specific audio playback"""
    
    def play(self, audio_path: Path) -> None:
        """Play audio using system-specific method"""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        system = platform.system().lower()
        
        try:
            if system == "windows":
                import winsound
                winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
            elif system == "darwin":  # macOS
                subprocess.run(["afplay", str(audio_path)], check=True)
            else:  # Linux or other
                subprocess.run(["aplay", str(audio_path)], check=True)
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            raise


class JupyterAudioPlayer:
    """Jupyter notebook audio playback"""
    
    def __init__(self):
        try:
            from IPython.display import display, Audio
            self.display = display
            self.Audio = Audio
            self.available = True
        except ImportError:
            self.available = False
    
    def play(self, audio_data: Any, sample_rate: int = 24000, autoplay: bool = True) -> None:
        """Display audio in Jupyter notebook"""
        if self.available:
            self.display(
                self.Audio(data=audio_data, rate=sample_rate, autoplay=autoplay),
                display_id=False
            )


class TTSEngine(ABC):
    """Abstract base for TTS engines"""
    
    @abstractmethod
    def synthesize(self, text: str, config: VoiceConfig) -> Generator:
        """Synthesize speech from text"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine is available"""
        pass


class KokoroTTSEngine(TTSEngine):
    """Kokoro TTS engine implementation"""
    
    LANG_CODES = {
        Language.ENGLISH: 'a',
        Language.CHINESE: 'z',
        Language.FRENCH: 'f',
        Language.JAPANESE: 'j'
    }
    
    VOICES = {
        Language.ENGLISH: ['af_kore', 'af_bella', 'af_alloy', 'af_nicole', 
                          'af_nova', 'af_sky', 'am_echo', 'am_michael', 'am_puck'],
        Language.CHINESE: ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
                          'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang'],
        Language.JAPANESE: ['jf_alpha', 'jf_gongitsune', 'jm_kumo'],
        Language.FRENCH: ['ff_siwis']
    }
    
    def __init__(self):
        self._pipeline = None
        self._available = False
        self._initialize()
    
    def _initialize(self):
        """Initialize the Kokoro pipeline"""
        try:
            from kokoro import KPipeline
            self._KPipeline = KPipeline
            self._available = True
        except ImportError:
            logger.warning("Kokoro package not found. TTS synthesis disabled.")
    
    def is_available(self) -> bool:
        return self._available
    
    def synthesize(self, text: str, config: VoiceConfig) -> Generator:
        """Synthesize speech using Kokoro"""
        if not self.is_available():
            raise RuntimeError("Kokoro TTS engine is not available")
        
        lang_code = self.LANG_CODES.get(config.language)
        if not lang_code:
            raise ValueError(f"Unsupported language: {config.language}")
        
        if self._pipeline is None:
            self._pipeline = self._KPipeline(lang_code=lang_code)
        
        return self._pipeline(
            text,
            voice=config.voice_name,
            speed=config.speed,
            split_pattern=r'\n+'
        )
    
    @classmethod
    def get_available_voices(cls, language: Language) -> List[str]:
        """Get available voices for a language"""
        return cls.VOICES.get(language, [])


class TTSService:
    """Main TTS service coordinating all components"""
    
    def __init__(
        self,
        engine: Optional[TTSEngine] = None,
        audio_player: Optional[AudioPlayer] = None,
        output_dir: Path = Path(".voices")
    ):
        self.engine = engine or KokoroTTSEngine()
        self.audio_player = audio_player or SystemAudioPlayer()
        self.jupyter_player = JupyterAudioPlayer()
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(exist_ok=True)
    
    def speak(
        self,
        text: str,
        language: Language = Language.ENGLISH,
        voice_index: int = 0,
        speed: float = 1.2
    ) -> Optional[Path]:
        """
        Convert text to speech and play it
        
        Returns:
            Path to the saved audio file, or None if synthesis failed
        """
        if not self.engine.is_available():
            logger.error("TTS engine is not available")
            return None
        
        # Get available voices
        voices = KokoroTTSEngine.get_available_voices(language)
        if not voices:
            logger.error(f"No voices available for {language}")
            return None
        
        voice_index = min(max(0, voice_index), len(voices) - 1)
        voice_name = voices[voice_index]
        
        processor = TextProcessorFactory.create(language)
        processed_text = processor.process(text)
        
        if not processed_text:
            logger.warning("No text to synthesize after processing")
            return None
        
        config = VoiceConfig(
            language=language,
            voice_name=voice_name,
            speed=speed
        )
        
        try:
            audio_file = self.output_dir / f"sample_{voice_name}.wav"
            for i, (_, _, audio) in enumerate(self.engine.synthesize(processed_text, config)):
                if self._is_jupyter():
                    self.jupyter_player.play(audio, autoplay=(i == 0))
                self._save_audio(audio_file, audio)
                self.audio_player.play(audio_file)
            return audio_file
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {e}")
            return None
    
    @staticmethod
    def _is_jupyter() -> bool:
        """Check if running in Jupyter environment"""
        from sys import modules
        return 'ipykernel' in modules
    
    @staticmethod
    def _save_audio(path: Path, audio_data: Any) -> None:
        """Save audio data to file"""
        try:
            import soundfile as sf
            sf.write(str(path), audio_data, 24000)
        except ImportError:
            logger.error("soundfile package not found. Cannot save audio.")
            raise


# Convenience functions for backward compatibility
def create_tts_service(enable: bool = True, output_dir: str = ".voices") -> Optional[TTSService]:
    """Create a TTS service instance"""
    if not enable:
        return None
    
    return TTSService(output_dir=Path(output_dir))


def speak(
    text: str,
    language: str = "en",
    voice_index: int = 0,
    speed: float = 1.2
) -> None:
    """Quick function to speak text"""
    service = create_tts_service()
    if service:
        try:
            lang_enum = Language(language)
            service.speak(text, lang_enum, voice_index, speed)
        except ValueError:
            logger.error(f"Invalid language: {language}")


if __name__ == "__main__":
    tts = create_tts_service()
    if tts:
        test_text = "Hello, I am ready to proceed with scientific experiments."
        tts.speak(test_text, Language.ENGLISH, voice_index=0)