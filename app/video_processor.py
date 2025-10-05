import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None
try:
    import av
except ImportError:
    av = None
import tempfile
import os
import math


class VideoProcessor:
    def __init__(self,
                 target_frames=134,
                 target_size=240,
                 sample_fps=10):
        self.target_frames = target_frames
        self.target_size = target_size
        self.sample_fps = sample_fps

    def extract_frames(self, video_path: str) -> np.ndarray:
        """Извлечение кадров из видео"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        # Получаем FPS видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Вычисляем шаг для выборки кадров
        frame_step = max(1, int(fps / self.sample_fps))

        frame_count = 0
        while cap.isOpened() and len(frames) < self.target_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                # Resize frame
                frame = cv2.resize(frame, (self.target_size, self.target_size))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            frame_count += 1

        cap.release()

        # Дополняем или обрезаем до target_frames
        frames = self._pad_or_crop_frames(frames)

        return np.array(frames)

    def _pad_or_crop_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Дополнение или обрезка кадров до нужного количества"""
        num_frames = len(frames)

        if num_frames < self.target_frames:
            # Дублируем последний кадр
            last_frame = frames[-1] if frames else np.zeros((self.target_size, self.target_size, 3))
            while len(frames) < self.target_frames:
                frames.append(last_frame)
        elif num_frames > self.target_frames:
            # Равномерно выбираем кадры
            indices = np.linspace(0, num_frames - 1, self.target_frames).astype(int)
            frames = [frames[i] for i in indices]

        return frames

    def extract_audio_features(self, video_path: str) -> dict:
        """Извлечение аудио характеристик"""
        if VideoFileClip is None:
            return {'has_audio': False, 'duration': 0}

        try:
            video = VideoFileClip(video_path)
            audio = video.audio

            if audio is not None:
                # Получаем аудио как numpy массив
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                audio.write_audiofile(temp_audio.name, logger=None)

                # Здесь можно добавить извлечение аудио-фич
                # Например, через librosa
                audio_features = {
                    'has_audio': True,
                    'duration': video.duration,
                    # Добавьте другие фичи при необходимости
                }

                os.unlink(temp_audio.name)
                video.close()

                return audio_features
            else:
                video.close()
                return {'has_audio': False, 'duration': 0}
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return {'has_audio': False, 'duration': 0}

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """Полная предобработка видео для модели"""
        # Извлекаем кадры
        frames = self.extract_frames(video_path)

        # Нормализуем значения пикселей
        frames = frames.astype(np.float32) / 255.0

        # Переставляем размерности: (T, H, W, C) -> (C, T, H, W)
        frames = np.transpose(frames, (3, 0, 1, 2))

        # Конвертируем в torch tensor
        tensor = torch.from_numpy(frames).float()

        # Добавляем batch dimension
        tensor = tensor.unsqueeze(0)  # (1, C, T, H, W)

        # Переставляем для модели: (B, T, C, H, W)
        tensor = tensor.permute(0, 2, 1, 3, 4)

        return tensor

    def extract_frames_fast(
        self,
        video_path: str,
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Быстрое извлечение кадров с использованием PyAV (в 3-5x быстрее OpenCV).
        Поддерживает seek к нужному моменту времени.

        Args:
            video_path: Путь к видеофайлу
            start_time: Начальная позиция в секундах
            duration: Длительность извлечения в секундах (None = до конца)

        Returns:
            np.ndarray: Массив кадров shape (target_frames, H, W, 3)
        """
        if av is None:
            # Fallback to OpenCV if PyAV not available
            print("PyAV not available, falling back to OpenCV")
            return self.extract_frames(video_path)

        frames = []

        try:
            container = av.open(video_path)
            stream = container.streams.video[0]

            # Получаем FPS
            fps = float(stream.average_rate)

            # Вычисляем временные границы
            start_pts = int(start_time * stream.time_base.denominator / stream.time_base.numerator)

            # Seek к нужной позиции
            if start_time > 0:
                container.seek(start_pts, stream=stream)

            # Вычисляем шаг для сэмплирования
            frame_step = max(1, int(fps / self.sample_fps))

            # Вычисляем конечное время
            end_time = start_time + duration if duration else float('inf')

            frame_count = 0
            extracted_count = 0

            for frame in container.decode(video=0):
                # Проверяем временные границы
                current_time = float(frame.pts * stream.time_base)

                if current_time < start_time:
                    continue

                if current_time > end_time:
                    break

                # Берем каждый N-й кадр
                if frame_count % frame_step == 0:
                    # Конвертируем в numpy array (RGB)
                    img = frame.to_ndarray(format='rgb24')

                    # Resize
                    img = cv2.resize(img, (self.target_size, self.target_size))

                    frames.append(img)
                    extracted_count += 1

                    # Прерываем если набрали достаточно кадров
                    if extracted_count >= self.target_frames:
                        break

                frame_count += 1

            container.close()

        except Exception as e:
            print(f"Error in PyAV extraction: {e}, falling back to OpenCV")
            return self.extract_frames(video_path)

        # Дополняем или обрезаем до target_frames
        frames = self._pad_or_crop_frames(frames)

        return np.array(frames)

    def get_video_duration(self, video_path: str) -> float:
        """
        Получить длительность видео в секундах.

        Args:
            video_path: Путь к видеофайлу

        Returns:
            float: Длительность в секундах
        """
        if av is not None:
            try:
                container = av.open(video_path)
                stream = container.streams.video[0]
                duration = float(stream.duration * stream.time_base)
                container.close()
                return duration
            except Exception as e:
                print(f"Error getting duration with PyAV: {e}")

        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
