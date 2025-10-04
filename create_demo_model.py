# virality-prediction/create_demo_model.py
import sys
import os

sys.path.append('/app')

import torch
from app.model import ViViT


def create_demo_model():
    """Создает демо-модель для тестирования"""
    print("Создание демо-модели...")

    model = ViViT(
        image_size=240,
        patch_size=16,
        num_classes=2,
        num_frames=134,
        dim=128
    )

    # Создаем случайные веса
    model_path = '/app/models/vivit_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Демо-модель сохранена: {model_path}")

    # Проверяем
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Размер модели: {size:.2f} MB")
        return True
    return False


if __name__ == "__main__":
    success = create_demo_model()
    exit(0 if success else 1)
