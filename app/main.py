from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil
from .predictor import ViralityPredictor
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(
    title="TikTok Virality Prediction API",
    description="API для предсказания вирусности видео на основе ViViT модели",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация предиктора
predictor = ViralityPredictor(model_path="/app/models/vivit_model.pth")


@app.get("/")
async def root():
    return {
        "message": "TikTok Virality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(predictor.device)
    }


@app.post("/predict")
async def predict_virality(file: UploadFile = File(...)):
    """
    Предсказание вирусности загруженного видео

    Returns:
        - virality_score: оценка от 0 до 100
        - is_viral: булево значение (True если вирусное)
        - probabilities: вероятности для каждого класса
        - confidence: уверенность модели
        - recommendations: список рекомендаций
    """

    # Проверка типа файла
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
        )

    # Проверка размера файла (макс 100MB)
    max_size = 100 * 1024 * 1024  # 100MB

    # Сохранение временного файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        try:
            # Копируем загруженный файл
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name

            # Проверка размера
            file_size = os.path.getsize(tmp_file_path)
            if file_size > max_size:
                os.unlink(tmp_file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл слишком большой. Максимальный размер: 100MB"
                )

            logger.info(f"Processing video: {file.filename} ({file_size / 1024 / 1024:.2f}MB)")

            # Предсказание
            result = predictor.predict(tmp_file_path)

            # Удаляем временный файл
            os.unlink(tmp_file_path)

            if result['success']:
                logger.info(f"Prediction complete: score={result['virality_score']}")
                return JSONResponse(content=result)
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ошибка обработки видео: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            # Убедимся, что временный файл удален
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

            logger.error(f"Error processing video: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при обработке видео: {str(e)}"
            )


@app.post("/predict_url")
async def predict_from_url(video_url: str):
    """
    Предсказание вирусности видео по URL (для интеграции с n8n)
    """
    # Здесь можно добавить логику скачивания видео по URL
    # и последующую обработку
    return {
        "message": "Эндпоинт для предсказания по URL",
        "url": video_url,
        "note": "Требует дополнительной реализации"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
