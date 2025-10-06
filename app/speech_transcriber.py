import whisper
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechTranscriber:
    """
    Транскрибирует речь из видео используя OpenAI Whisper.
    """

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: Размер модели Whisper (tiny, base, small, medium, large)
                       base - хороший баланс скорости и качества
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Whisper ({model_size}) on device: {self.device}")

        try:
            self.model = whisper.load_model(model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(self, video_path: str, language: Optional[str] = None) -> str:
        """
        Транскрибирует речь из видео.

        Args:
            video_path: Путь к видео файлу
            language: Код языка (ru, en, etc.). Если None - автоопределение

        Returns:
            Текст транскрипции
        """
        try:
            logger.info(f"Transcribing video: {video_path}")

            # Whisper автоматически извлекает аудио из видео
            result = self.model.transcribe(
                video_path,
                language=language,
                fp16=(self.device == "cuda"),  # FP16 только на GPU
                verbose=False
            )

            text = result["text"].strip()
            detected_language = result.get("language", "unknown")

            logger.info(f"Transcription complete: {len(text)} chars, language: {detected_language}")

            return text

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def transcribe_with_timestamps(self, video_path: str, language: Optional[str] = None) -> list:
        """
        Транскрибирует речь с временными метками.

        Args:
            video_path: Путь к видео файлу
            language: Код языка (ru, en, etc.)

        Returns:
            Список сегментов с текстом и временными метками
            [{"start": 0.0, "end": 5.2, "text": "Hello"}, ...]
        """
        try:
            logger.info(f"Transcribing with timestamps: {video_path}")

            result = self.model.transcribe(
                video_path,
                language=language,
                fp16=(self.device == "cuda"),
                verbose=False
            )

            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })

            logger.info(f"Transcription complete: {len(segments)} segments")

            return segments

        except Exception as e:
            logger.error(f"Transcription with timestamps failed: {e}")
            return []
