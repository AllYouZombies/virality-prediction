from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
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


# Pydantic models для запросов
class AnalyzeTimelineRequest(BaseModel):
    window_duration: float = 60.0
    stride: float = 55.0
    offset_b: float = 30.0
    top_n: int = 15
    batch_size: int = 4
    min_score: Optional[int] = None

class AnalyzeTimelinePathRequest(BaseModel):
    video_path: str
    window_duration: float = 60.0
    stride: float = 55.0
    offset_b: float = 30.0
    top_n: int = 15
    batch_size: int = 4
    min_score: Optional[int] = None


@app.get("/")
async def root():
    return {
        "message": "TikTok Virality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "analyze_timeline": "/analyze_timeline",
            "analyze_timeline_path": "/analyze_timeline_path",
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


@app.post("/analyze_timeline")
async def analyze_timeline(
    file: UploadFile = File(...),
    window_duration: float = 60.0,
    stride: float = 55.0,
    offset_b: float = 30.0,
    top_n: int = 15,
    batch_size: int = 4,
    min_score: Optional[int] = None
):
    """
    Анализ всего видеофайла с помощью sliding window.
    Возвращает timeline со скорами вирусности для всех окон и топ-N фрагментов.

    Args:
        file: Видеофайл для анализа
        window_duration: Длительность окна в секундах (по умолчанию 60)
        stride: Шаг окна в секундах (по умолчанию 55, overlap 5 сек)
        offset_b: Сдвиг для второго набора окон в секундах (по умолчанию 30)
        top_n: Количество топ-фрагментов для возврата (по умолчанию 15)
        batch_size: Размер батча для GPU (по умолчанию 4)
        min_score: Минимальный score для включения в результаты (опционально)

    Returns:
        - total_windows: Общее количество проанализированных окон
        - total_duration: Длительность видео в секундах
        - top_segments: Топ-N фрагментов с высокими скорами
        - all_scores: Все проанализированные окна (опционально)
    """
    # Проверка типа файла
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
        )

    # Сохранение временного файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file_path = None
        try:
            # Копируем загруженный файл
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name

            logger.info(f"Analyzing timeline for: {file.filename}")

            # Получаем длительность видео
            duration = predictor.video_processor.get_video_duration(tmp_file_path)
            logger.info(f"Video duration: {duration:.2f} seconds")

            # Создаем окна для набора A
            windows_a = []
            start = 0
            while start < duration:
                actual_duration = min(window_duration, duration - start)
                if actual_duration > 5:  # Игнорируем слишком короткие фрагменты
                    windows_a.append({
                        'start': start,
                        'duration': actual_duration,
                        'set': 'A',
                        'index': len(windows_a)
                    })
                start += stride

            # Создаем окна для набора B (со сдвигом)
            windows_b = []
            start = offset_b
            while start < duration:
                actual_duration = min(window_duration, duration - start)
                if actual_duration > 5:
                    windows_b.append({
                        'start': start,
                        'duration': actual_duration,
                        'set': 'B',
                        'index': len(windows_b)
                    })
                start += stride

            all_windows = windows_a + windows_b
            logger.info(f"Created {len(windows_a)} windows for set A, {len(windows_b)} for set B")

            # Извлекаем кадры для всех окон
            logger.info("Extracting frames for all windows...")
            all_frames = []
            for window in all_windows:
                frames = predictor.video_processor.extract_frames_fast(
                    tmp_file_path,
                    start_time=window['start'],
                    duration=window['duration']
                )
                all_frames.append(frames)

            # Батч-предсказание
            logger.info(f"Running batch prediction with batch_size={batch_size}...")
            predictions = predictor.predict_batch(all_frames, batch_size=batch_size)

            # Формируем результаты
            results = []
            for window, prediction in zip(all_windows, predictions):
                if prediction['success']:
                    result = {
                        'start': window['start'],
                        'end': window['start'] + window['duration'],
                        'duration': window['duration'],
                        'set': window['set'],
                        'index': window['index'],
                        'score': prediction['virality_score'],
                        'is_viral': prediction['is_viral'],
                        'confidence': prediction['confidence'],
                        'probabilities': prediction['probabilities']
                    }

                    # Фильтруем по min_score если указан
                    if min_score is None or result['score'] >= min_score:
                        results.append(result)
                else:
                    logger.warning(f"Failed prediction for window at {window['start']}s: {prediction.get('error')}")

            # Сортируем по score и берем топ-N
            top_segments = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

            # Удаляем временный файл
            os.unlink(tmp_file_path)

            logger.info(f"Analysis complete. Found {len(results)} segments, returning top {len(top_segments)}")

            return JSONResponse(content={
                'success': True,
                'filename': file.filename,
                'total_duration': duration,
                'total_windows': len(all_windows),
                'windows_analyzed': len(results),
                'windows_a': len(windows_a),
                'windows_b': len(windows_b),
                'top_segments': top_segments,
                'parameters': {
                    'window_duration': window_duration,
                    'stride': stride,
                    'offset_b': offset_b,
                    'batch_size': batch_size,
                    'min_score': min_score
                }
            })

        except Exception as e:
            # Убедимся, что временный файл удален
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

            logger.error(f"Error analyzing timeline: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при анализе timeline: {str(e)}"
            )


@app.post("/analyze_timeline_path")
async def analyze_timeline_path(request: AnalyzeTimelinePathRequest):
    """
    Анализ видеофайла по пути (для интеграции с n8n через shared volume).
    Работает аналогично /analyze_timeline, но принимает путь к файлу вместо загрузки.

    Args:
        request: JSON body с параметрами анализа
            - video_path: Путь к видеофайлу на shared volume (например, /files/video.mp4)
            - window_duration: Длительность окна в секундах (по умолчанию 60)
            - stride: Шаг окна в секундах (по умолчанию 55, overlap 5 сек)
            - offset_b: Сдвиг для второго набора окон в секундах (по умолчанию 30)
            - top_n: Количество топ-фрагментов для возврата (по умолчанию 15)
            - batch_size: Размер батча для GPU (по умолчанию 4)
            - min_score: Минимальный score для включения в результаты (опционально)

    Returns:
        JSON с топ-N фрагментами и их временными метками
    """
    video_path = request.video_path
    window_duration = request.window_duration
    stride = request.stride
    offset_b = request.offset_b
    top_n = request.top_n
    batch_size = request.batch_size
    min_score = request.min_score

    # Проверка существования файла
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Файл не найден: {video_path}"
        )

    # Проверка типа файла
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(video_path)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
        )

    try:
        logger.info(f"Analyzing timeline for: {video_path}")

        # Получаем длительность видео
        duration = predictor.video_processor.get_video_duration(video_path)
        logger.info(f"Video duration: {duration:.2f} seconds")

        # Создаем окна для набора A
        windows_a = []
        start = 0
        while start < duration:
            actual_duration = min(window_duration, duration - start)
            if actual_duration > 5:  # Игнорируем слишком короткие фрагменты
                windows_a.append({
                    'start': start,
                    'duration': actual_duration,
                    'set': 'A',
                    'index': len(windows_a)
                })
            start += stride

        # Создаем окна для набора B (со сдвигом)
        windows_b = []
        start = offset_b
        while start < duration:
            actual_duration = min(window_duration, duration - start)
            if actual_duration > 5:
                windows_b.append({
                    'start': start,
                    'duration': actual_duration,
                    'set': 'B',
                    'index': len(windows_b)
                })
            start += stride

        all_windows = windows_a + windows_b
        logger.info(f"Created {len(windows_a)} windows for set A, {len(windows_b)} for set B")

        # Извлекаем кадры для всех окон
        logger.info("Extracting frames for all windows...")
        all_frames = []
        for window in all_windows:
            frames = predictor.video_processor.extract_frames_fast(
                video_path,
                start_time=window['start'],
                duration=window['duration']
            )
            all_frames.append(frames)

        # Батч-предсказание
        logger.info(f"Running batch prediction with batch_size={batch_size}...")
        predictions = predictor.predict_batch(all_frames, batch_size=batch_size)

        # Формируем результаты
        results = []
        for window, prediction in zip(all_windows, predictions):
            if prediction['success']:
                result = {
                    'start': window['start'],
                    'end': window['start'] + window['duration'],
                    'duration': window['duration'],
                    'set': window['set'],
                    'index': window['index'],
                    'score': prediction['virality_score'],
                    'is_viral': prediction['is_viral'],
                    'confidence': prediction['confidence'],
                    'probabilities': prediction['probabilities']
                }

                # Фильтруем по min_score если указан
                if min_score is None or result['score'] >= min_score:
                    results.append(result)
            else:
                logger.warning(f"Failed prediction for window at {window['start']}s: {prediction.get('error')}")

        # Сортируем по score и берем топ-N
        top_segments = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

        logger.info(f"Analysis complete. Found {len(results)} segments, returning top {len(top_segments)}")

        return JSONResponse(content={
            'success': True,
            'video_path': video_path,
            'total_duration': duration,
            'total_windows': len(all_windows),
            'windows_analyzed': len(results),
            'windows_a': len(windows_a),
            'windows_b': len(windows_b),
            'top_segments': top_segments,
            'parameters': {
                'window_duration': window_duration,
                'stride': stride,
                'offset_b': offset_b,
                'batch_size': batch_size,
                'min_score': min_score
            }
        })

    except Exception as e:
        logger.error(f"Error analyzing timeline: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при анализе timeline: {str(e)}"
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
