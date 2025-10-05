import torch
import torchaudio
import numpy as np
from typing import List, Dict, Tuple
import tempfile
import os
import logging

logger = logging.getLogger(__name__)


class SpeechDetector:
    """Детектор речи на основе Silero VAD"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Silero VAD on device: {self.device}")

        # Загружаем Silero VAD модель
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.model.to(self.device)

            # Получаем утилиты
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils

            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def extract_audio_from_video(self, video_path: str) -> Tuple[torch.Tensor, int]:
        """
        Извлекает аудио из видео файла.

        Returns:
            Tuple[torch.Tensor, int]: (audio_tensor, sample_rate)
        """
        import subprocess
        import soundfile as sf

        # Создаем временный wav файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio_path = tmp_audio.name

        try:
            # Извлекаем аудио с помощью ffmpeg
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # без видео
                '-acodec', 'pcm_s16le',  # WAV формат
                '-ar', '16000',  # 16kHz (требование Silero)
                '-ac', '1',  # моно
                '-y',  # перезаписать
                tmp_audio_path
            ]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Загружаем аудио напрямую через soundfile (избегаем torchcodec)
            audio_data, sr = sf.read(tmp_audio_path, dtype='float32')

            # Конвертируем в torch тензор
            wav = torch.from_numpy(audio_data)

            # Если sample rate не 16000 - ресэмплируем
            if sr != 16000:
                # Используем torchaudio для ресэмплинга
                wav = wav.unsqueeze(0)  # добавляем канал
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
                wav = wav.squeeze(0)
                sr = 16000

            return wav, sr

        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)

    def detect_speech_segments(
        self,
        video_path: str,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30
    ) -> List[Dict[str, float]]:
        """
        Детектирует сегменты с речью в видео.

        Args:
            video_path: Путь к видео файлу
            threshold: Порог вероятности речи (0-1)
            min_speech_duration_ms: Минимальная длительность речи в мс
            min_silence_duration_ms: Минимальная длительность тишины в мс
            window_size_samples: Размер окна для анализа
            speech_pad_ms: Padding вокруг речи в мс

        Returns:
            List[Dict]: Список сегментов вида [{'start': 1.5, 'end': 3.2}, ...]
        """
        try:
            # Извлекаем аудио
            wav, sr = self.extract_audio_from_video(video_path)
            wav = wav.to(self.device)

            # Получаем временные метки речи
            speech_timestamps = self.get_speech_timestamps(
                wav,
                self.model,
                threshold=threshold,
                sampling_rate=sr,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                window_size_samples=window_size_samples,
                speech_pad_ms=speech_pad_ms
            )

            # Конвертируем из сэмплов в секунды
            segments = []
            for ts in speech_timestamps:
                segments.append({
                    'start': ts['start'] / sr,
                    'end': ts['end'] / sr
                })

            logger.info(f"Detected {len(segments)} speech segments in {video_path}")
            return segments

        except Exception as e:
            logger.error(f"Error detecting speech: {e}")
            # В случае ошибки возвращаем пустой список (будет работать как отсутствие речи)
            return []

    def has_speech_near(
        self,
        speech_segments: List[Dict[str, float]],
        timestamp: float,
        tolerance: float = 0.5
    ) -> bool:
        """
        Проверяет есть ли речь около указанного времени.

        Args:
            speech_segments: Список сегментов речи
            timestamp: Проверяемая временная метка в секундах
            tolerance: Допуск в секундах

        Returns:
            bool: True если есть речь в пределах tolerance
        """
        for segment in speech_segments:
            # Проверяем пересечение с учетом tolerance
            if (segment['start'] - tolerance <= timestamp <= segment['end'] + tolerance):
                return True
        return False

    def has_speech_in_range(
        self,
        speech_segments: List[Dict[str, float]],
        start: float,
        end: float
    ) -> bool:
        """
        Проверяет есть ли речь в указанном диапазоне.

        Args:
            speech_segments: Список сегментов речи
            start: Начало диапазона в секундах
            end: Конец диапазона в секундах

        Returns:
            bool: True если есть речь в диапазоне
        """
        for segment in speech_segments:
            # Проверяем любое пересечение
            if not (segment['end'] < start or segment['start'] > end):
                return True
        return False

    def merge_close_segments(
        self,
        segments: List[Dict[str, float]],
        max_gap: float = 0.5
    ) -> List[Dict[str, float]]:
        """
        Объединяет близкие сегменты речи.

        Args:
            segments: Список сегментов
            max_gap: Максимальный разрыв для объединения в секундах

        Returns:
            List[Dict]: Объединенные сегменты
        """
        if not segments:
            return []

        # Сортируем по начальному времени
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        merged = [sorted_segments[0].copy()]

        for current in sorted_segments[1:]:
            last = merged[-1]

            # Если текущий сегмент близко к последнему - объединяем
            if current['start'] - last['end'] <= max_gap:
                last['end'] = max(last['end'], current['end'])
            else:
                merged.append(current.copy())

        return merged
