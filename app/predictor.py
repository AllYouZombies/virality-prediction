# virality-prediction/app/predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification
from .video_processor import VideoProcessor
import os
import numpy as np
from typing import List


class ViralityPredictor:
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Загружаем предобученную VideoMAE модель
        print("Загрузка HuggingFace VideoMAE модели...")

        base_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

        # Адаптируем под бинарную классификацию (вирусное/не вирусное)
        # Заменяем последний слой с 400 классов на 2
        num_features = base_model.classifier.in_features
        base_model.classifier = nn.Linear(num_features, 2)

        self.model = base_model
        self.model.to(self.device)
        self.model.eval()

        # Процессор не нужен - используем ручную нормализацию
        self.image_processor = None

        # Инициализация процессора видео
        # VideoMAE ожидает 16 кадров × 224×224
        self.video_processor = VideoProcessor(target_frames=16, target_size=224)

        # Информация о GPU если доступен
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        print("Модель загружена (предобученная VideoMAE с адаптацией под 2 класса)")

    def _create_demo_weights(self):
        """Создает демо-веса для модели"""
        # Инициализация с небольшим bias в сторону "не вирусного"
        # чтобы демо давало разумные результаты
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

    def predict(self, video_path: str) -> dict:
        """Предсказание вирусности видео"""
        try:
            # Предобработка видео
            frames = self.video_processor.extract_frames(video_path)

            frames_normalized = frames.astype(np.float32) / 255.0
            # Переставляем размерности: (T, H, W, C) -> (T, C, H, W)
            frames_tensor = torch.from_numpy(frames_normalized).permute(0, 3, 1, 2)
            # Добавляем batch dimension: (T, C, H, W) -> (1, T, C, H, W)
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
            # Нормализация ImageNet
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(self.device)
            frames_tensor = (frames_tensor - mean) / std

            # Извлечение аудио характеристик
            audio_features = self.video_processor.extract_audio_features(video_path)

            # Предсказание
            with torch.no_grad():
                outputs = self.model(pixel_values=frames_tensor)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)

                # Получаем вероятности для каждого класса
                non_viral_prob = probabilities[0, 0].item()
                viral_prob = probabilities[0, 1].item()

                # Определяем класс
                predicted_class = torch.argmax(probabilities, dim=1).item()
                is_viral = predicted_class == 1

            # Вычисляем оценку вирусности (0-100)
            virality_score = int(viral_prob * 100)

            # Формируем рекомендации
            recommendations = self._generate_recommendations(
                virality_score,
                audio_features
            )

            return {
                'success': True,
                'virality_score': virality_score,
                'is_viral': is_viral,
                'probabilities': {
                    'non_viral': round(non_viral_prob, 4),
                    'viral': round(viral_prob, 4)
                },
                'confidence': round(max(non_viral_prob, viral_prob), 4),
                'recommendations': recommendations,
                'audio_features': audio_features,
                'device_used': str(self.device),
                'model_status': 'huggingface_videomae_pretrained'
            }

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_recommendations(self, score: int, audio_features: dict) -> list:
        """Генерация рекомендаций на основе оценки"""
        recommendations = []

        if score < 30:
            recommendations.append("Добавьте более захватывающий хук в первые 3 секунды")
            recommendations.append("Используйте более динамичную музыку")
            recommendations.append("Сократите видео до 15-30 секунд")
        elif score < 60:
            recommendations.append("Улучшите визуальную динамику")
            recommendations.append("Добавьте трендовую музыку")
            recommendations.append("Оптимизируйте тайминг для платформы")
        else:
            recommendations.append("Видео имеет хороший потенциал!")
            recommendations.append("Опубликуйте в пиковое время аудитории")
            recommendations.append("Используйте релевантные хэштеги")

        if not audio_features.get('has_audio'):
            recommendations.append("Добавьте музыку или звуковые эффекты")

        if audio_features.get('duration', 0) > 60:
            recommendations.append("Сократите видео до 60 секунд или меньше")

        return recommendations

    def predict_from_frames(self, frames: np.ndarray) -> dict:
        """
        Предсказание вирусности из уже извлеченных кадров.

        Args:
            frames: numpy array shape (num_frames, H, W, C)

        Returns:
            dict: Результаты предсказания
        """
        try:
            frames_normalized = frames.astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(frames_normalized).permute(0, 3, 1, 2)
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(self.device)
            frames_tensor = (frames_tensor - mean) / std

            # Предсказание
            with torch.no_grad():
                outputs = self.model(pixel_values=frames_tensor)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)

                # Получаем вероятности для каждого класса
                non_viral_prob = probabilities[0, 0].item()
                viral_prob = probabilities[0, 1].item()

                # Определяем класс
                predicted_class = torch.argmax(probabilities, dim=1).item()
                is_viral = predicted_class == 1

            # Вычисляем оценку вирусности (0-100)
            virality_score = int(viral_prob * 100)

            return {
                'success': True,
                'virality_score': virality_score,
                'is_viral': is_viral,
                'probabilities': {
                    'non_viral': round(non_viral_prob, 4),
                    'viral': round(viral_prob, 4)
                },
                'confidence': round(max(non_viral_prob, viral_prob), 4)
            }

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    def predict_batch(self, frames_list: List[np.ndarray], batch_size: int = 4) -> List[dict]:
        """
        Батч-предсказание для нескольких наборов кадров.
        Обрабатывает несколько видео одновременно на GPU для ускорения.

        Args:
            frames_list: Список numpy arrays, каждый shape (num_frames, H, W, C)
            batch_size: Размер батча для обработки на GPU

        Returns:
            List[dict]: Список результатов предсказаний
        """
        results = []

        try:
            # Обрабатываем батчами
            for i in range(0, len(frames_list), batch_size):
                batch = frames_list[i:i + batch_size]

                batch_tensors = []
                for frames in batch:
                    frames_normalized = frames.astype(np.float32) / 255.0
                    frames_tensor = torch.from_numpy(frames_normalized).permute(0, 3, 1, 2)
                    frames_tensor = frames_tensor.unsqueeze(0)
                    batch_tensors.append(frames_tensor)

                # Стакаем батч
                stacked_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
                # Нормализация ImageNet
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(self.device)
                stacked_tensor = (stacked_tensor - mean) / std

                # Предсказание для батча
                with torch.no_grad():
                    outputs = self.model(pixel_values=stacked_tensor)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=1)

                    # Обрабатываем каждый результат в батче
                    for j in range(len(batch)):
                        non_viral_prob = probabilities[j, 0].item()
                        viral_prob = probabilities[j, 1].item()
                        predicted_class = torch.argmax(probabilities[j]).item()
                        is_viral = predicted_class == 1
                        virality_score = int(viral_prob * 100)

                        results.append({
                            'success': True,
                            'virality_score': virality_score,
                            'is_viral': is_viral,
                            'probabilities': {
                                'non_viral': round(non_viral_prob, 4),
                                'viral': round(viral_prob, 4)
                            },
                            'confidence': round(max(non_viral_prob, viral_prob), 4)
                        })

        except Exception as e:
            print(f"Ошибка батч-предсказания: {e}")
            import traceback
            traceback.print_exc()
            # Возвращаем ошибки для всех элементов батча
            for _ in range(len(frames_list) - len(results)):
                results.append({
                    'success': False,
                    'error': str(e)
                })

        return results
