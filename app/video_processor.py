import cv2
import numpy as np
import torch
from typing import Tuple, List
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None
import tempfile
import os


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
