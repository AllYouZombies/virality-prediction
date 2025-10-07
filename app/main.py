from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os
import shutil
from .predictor import ViralityPredictor
from .speech_detector import SpeechDetector
from .speech_transcriber import SpeechTranscriber
from .trimming_optimizer_v2 import TrimmingOptimizerV2
import logging
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import requests
import base64
import cv2
from PIL import Image
import io

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(
    title="TikTok Virality Prediction API",
    description="API для предсказания вирусности видео на основе ML",
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
predictor = ViralityPredictor(model_path="/app/models/model.pth")

# Инициализация детектора речи, транскрибера и оптимизатора
speech_detector = SpeechDetector()
speech_transcriber = SpeechTranscriber(model_size="base")  # base - баланс скорости и качества

trimming_optimizer = TrimmingOptimizerV2(
    target_duration=60.0,
    step_size=10.0,  # Размер шага для анализа
    speech_gap_tolerance=0.3  # 300мс - минимальная пауза для обрезки
)


def get_optimal_batch_size() -> int:
    """
    Автоматически определяет оптимальный batch_size на основе доступных ресурсов.
    Максимизирует использование GPU памяти.
    """
    if torch.cuda.is_available():
        # Получаем доступную GPU память
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        gpu_mem_free = (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_allocated(0)) / 1e9  # GB

        # VideoMAE требует ~500MB на batch из 16 видео (16 кадров, 224x224)
        # Используем 80% свободной памяти для безопасности
        mem_per_video = 0.03  # GB (~30MB на одно видео)
        optimal_batch = int((gpu_mem_free * 0.8) / mem_per_video)

        # Ограничиваем разумными пределами
        optimal_batch = max(32, min(optimal_batch, 512))

        logger.info(f"GPU Memory: {gpu_mem_free:.2f}GB free / {gpu_mem_total:.2f}GB total")
        logger.info(f"Optimal batch size: {optimal_batch}")

        return optimal_batch
    else:
        # CPU режим - используем все ядра
        cpu_count = multiprocessing.cpu_count()
        optimal_batch = max(4, cpu_count * 2)

        logger.info(f"CPU cores: {cpu_count}, Optimal batch size: {optimal_batch}")

        return optimal_batch


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

class AnalyzeAndOptimizeRequest(BaseModel):
    video_path: str
    window_duration: float = 180.0  # Размер блоков для анализа (по умолчанию 3 минуты)
    overlap: float = 10.0  # Наложение между блоками (по умолчанию 10 секунд)
    target_duration: float = 60.0  # Выходная длительность нарезок (по умолчанию 60 секунд)
    top_n: int = 10  # Количество выходных нарезок (по умолчанию 10 штук)
    micro_window: float = 15.0  # Шаг анализа внутри блоков (по умолчанию 15 секунд)
    speech_gap_tolerance: float = 0.3  # Минимальная пауза речи для обрезки (по умолчанию 300мс)

    # Автоматические параметры (не настраиваются пользователем)
    batch_size: Optional[int] = None  # Автоматически определяется по GPU памяти
    enable_speech_detection: bool = True


@app.get("/")
async def root():
    return {
        "message": "TikTok Virality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "analyze_timeline": "/analyze_timeline",
            "analyze_timeline_path": "/analyze_timeline_path",
            "analyze_and_optimize": "/analyze_and_optimize",
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

        # Батч-предсказание с VideoMAE (быстрая оценка)
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


@app.post("/analyze_and_optimize")
async def analyze_and_optimize(request: AnalyzeAndOptimizeRequest):
    """
    Анализирует видео окнами по 2 минуты, делает детальный анализ по 5 сек,
    и оптимизирует до 60 секунд вырезая наименее вирусные фрагменты.

    Args:
        request: Параметры анализа и оптимизации

    Returns:
        JSON с оптимизированными фрагментами и командами FFmpeg
    """
    video_path = request.video_path
    window_duration = request.window_duration
    overlap = request.overlap
    stride = window_duration - overlap  # Автоматический расчет шага
    micro_window = request.micro_window
    top_n = request.top_n

    # Автоматическое определение оптимального batch_size
    batch_size = request.batch_size or get_optimal_batch_size()
    logger.info(f"Using batch_size={batch_size} for GPU/CPU processing")

    # Проверка существования файла
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Файл не найден: {video_path}"
        )

    try:
        logger.info(f"Analyzing and optimizing: {video_path}")

        # Создаем оптимизатор с параметрами из запроса
        # step_size = micro_window (шаг анализа = шаг группировки)
        optimizer = TrimmingOptimizerV2(
            target_duration=request.target_duration,
            step_size=request.micro_window,
            speech_gap_tolerance=request.speech_gap_tolerance
        )

        # Шаг 1: Детекция речи (если включено)
        speech_segments = []
        if request.enable_speech_detection and speech_detector:
            logger.info("Detecting speech segments...")
            speech_segments = speech_detector.detect_speech_segments(video_path)
            logger.info(f"Found {len(speech_segments)} speech segments")

        # Шаг 2: Получаем длительность видео
        duration = predictor.video_processor.get_video_duration(video_path)
        logger.info(f"Video duration: {duration:.2f} seconds")

        # Шаг 3: Создаем окна с перекрытием
        all_windows = []
        start = 0
        while start < duration:
            actual_duration = min(window_duration, duration - start)
            if actual_duration > 10:  # Игнорируем слишком короткие фрагменты
                all_windows.append({
                    'start': start,
                    'duration': actual_duration,
                    'index': len(all_windows)
                })
            start += stride

        logger.info(f"Created {len(all_windows)} windows (size={window_duration}s, overlap={overlap}s)")

        # Шаг 4: Параллельная экстракция кадров для всех окон (максимизируем I/O)
        logger.info(f"Starting parallel frame extraction for {len(all_windows)} windows...")

        def extract_window_frames(window):
            """Извлекает микро-сегменты и кадры для одного окна"""
            # 4.1: Создаем микро-окна внутри окна
            micro_segments = []
            micro_start = window['start']
            window_end = window['start'] + window['duration']

            while micro_start < window_end:
                micro_duration = min(micro_window, window_end - micro_start)
                if micro_duration >= 3:  # Минимум 3 секунды
                    micro_segments.append({
                        'start': micro_start,
                        'duration': micro_duration,
                        'end': micro_start + micro_duration
                    })
                micro_start += micro_window

            # 4.2: Извлекаем кадры для всех микро-окон
            all_frames = []
            for micro_seg in micro_segments:
                frames = predictor.video_processor.extract_frames_fast(
                    video_path,
                    start_time=micro_seg['start'],
                    duration=micro_seg['duration']
                )
                all_frames.append(frames)

            return {
                'window': window,
                'micro_segments': micro_segments,
                'all_frames': all_frames
            }

        # Параллельная экстракция кадров (используем все CPU ядра)
        cpu_workers = multiprocessing.cpu_count()
        with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            window_data_list = list(executor.map(extract_window_frames, all_windows))

        logger.info(f"Frame extraction complete, processing predictions on GPU...")

        # Шаг 5: Для каждого окна делаем предсказание и оптимизацию
        optimized_segments = []

        for window_data in window_data_list:
            window = window_data['window']
            micro_segments = window_data['micro_segments']
            all_frames = window_data['all_frames']

            # Батч-предсказание для микро-окон (кадры уже извлечены параллельно)
            predictions = predictor.predict_batch(all_frames, batch_size=batch_size)

            # 4.4: Добавляем scores к микро-сегментам
            for micro_seg, prediction in zip(micro_segments, predictions):
                if prediction['success']:
                    micro_seg['score'] = prediction['virality_score']
                    micro_seg['is_viral'] = prediction['is_viral']
                    micro_seg['confidence'] = prediction['confidence']
                else:
                    micro_seg['score'] = 0
                    micro_seg['is_viral'] = False
                    micro_seg['confidence'] = 0

            # 4.5: Оптимизируем окно до 60 секунд
            is_last_segment = window == all_windows[-1]
            optimization_result = optimizer.optimize_segment(
                micro_segments,
                speech_segments,
                window['start'],
                window['duration'],
                is_last_segment
            )

            # 4.6: Генерируем FFmpeg фильтры
            # keep_ranges уже в абсолютных координатах от optimize_segment
            ffmpeg_select_filter = optimizer.generate_ffmpeg_filter(
                optimization_result['keep_ranges']
            )
            ffmpeg_audio_filter = optimizer.generate_ffmpeg_audio_filter(
                optimization_result['keep_ranges']
            )

            logger.info(f"Window {window['index']}: keep_ranges={optimization_result['keep_ranges']}, filter='{ffmpeg_select_filter}'")

            # 4.8: Сохраняем результат с полями для воркфлоу
            optimized_segments.append({
                # Поля для воркфлоу (совместимость)
                'start': float(window['start']),
                'end': float(window['start'] + window['duration']),
                'duration': float(window['duration']),
                'score': int(optimization_result['avg_score']),
                'is_viral': bool(optimization_result['avg_score'] >= 50),
                'confidence': float(optimization_result['avg_score'] / 100.0),
                'index': window['index'],

                # Оптимизация
                'optimization': {
                    'optimized_duration': float(optimization_result['final_duration']),
                    'time_saved': float(optimization_result['removed_duration']),
                    'num_cuts': int(len(optimization_result['remove_ranges']))
                },

                # FFmpeg фильтры
                'ffmpeg_select_filter': ffmpeg_select_filter,
                'ffmpeg_audio_filter': ffmpeg_audio_filter,

                # Детальная информация
                'keep_ranges': [[float(s), float(e)] for s, e in optimization_result['keep_ranges']],
                'remove_ranges': [[float(s), float(e)] for s, e in optimization_result['remove_ranges']],
                'avg_score': float(optimization_result['avg_score'])
            })

        # Шаг 5: Сортируем по среднему score и берем топ-N
        top_segments = sorted(optimized_segments, key=lambda x: x['avg_score'], reverse=True)[:top_n]

        logger.info(f"Optimization complete. Returning top {len(top_segments)} segments")

        return JSONResponse(content={
            'success': True,
            'video_path': video_path,
            'total_duration': duration,
            'total_windows': len(all_windows),
            'speech_segments_detected': len(speech_segments),
            'top_segments': top_segments,
            'parameters': {
                'window_duration': window_duration,
                'overlap': overlap,
                'stride': stride,
                'micro_window': micro_window,
                'target_duration': request.target_duration,
                'speech_gap_tolerance': request.speech_gap_tolerance,
                'batch_size': batch_size,
                'enable_speech_detection': request.enable_speech_detection
            }
        })

    except Exception as e:
        logger.error(f"Error analyzing and optimizing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при анализе и оптимизации: {str(e)}"
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


class GenerateMetadataRequest(BaseModel):
    video_path: str
    virality_score: int
    duration: float
    output_path: str  # Путь к выходному MP4 файлу


@app.post("/generate_metadata")
async def generate_metadata(request: GenerateMetadataRequest):
    """
    Генерирует название и описание для YouTube Shorts используя Qwen2.5-VL через Ollama.
    Сохраняет два текстовых файла рядом с видео:
    - {output_path}_title.txt - название ролика
    - {output_path}_description.txt - описание с хештегами
    """
    try:
        video_path = request.video_path
        score = request.virality_score
        duration = request.duration
        output_path = request.output_path

        logger.info(f"Generating metadata for {output_path}, score={score}, duration={duration:.1f}s")

        # Извлекаем несколько ключевых кадров из ОБРАБОТАННОГО видео для полного анализа
        cap = cv2.VideoCapture(output_path)

        # Получаем общее количество кадров и FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / fps if fps > 0 else duration

        # Берем 4 ключевых момента: начало, 1/3, 2/3, конец
        frame_positions = [0, int(total_frames * 0.33), int(total_frames * 0.66), max(0, total_frames - 1)]

        frames_base64 = []
        max_size = 512

        for frame_idx in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Конвертируем кадр в base64
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Resize для экономии памяти (max 512px по длинной стороне)
                ratio = min(max_size / pil_image.width, max_size / pil_image.height)
                if ratio < 1:
                    new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG", quality=85)
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                frames_base64.append(img_base64)

        cap.release()

        if not frames_base64:
            raise HTTPException(status_code=400, detail="Failed to extract frames from processed video")

        logger.info(f"Extracted {len(frames_base64)} keyframes for analysis")

        # Транскрибируем речь из видео
        transcription = ""
        try:
            logger.info("Transcribing speech from video...")
            transcription = speech_transcriber.transcribe(output_path, language="ru")
            if transcription:
                logger.info(f"Transcription complete: {len(transcription)} chars")
            else:
                logger.warning("No speech detected in video")
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            transcription = ""

        # Формируем промпт для Ollama с транскрипцией
        speech_context = f"\n\nТЕКСТ ИЗ ВИДЕО:\n{transcription}" if transcription else ""

        prompt = f"""Ты эксперт по вирусному контенту YouTube Shorts. Тебе даны 4 ключевых кадра из видео (начало, треть, две трети, конец){' и текст речи из видео' if transcription else ''}. Проанализируй весь контент видео и сгенерируй:

1. ВИРУСНОЕ НАЗВАНИЕ (до 100 символов):
   - Должно цеплять внимание
   - Использовать триггерные слова
   - НЕ использовать emoji
   - Отражать ключевые моменты из всех кадров

2. ОПИСАНИЕ С ХЭШТЕГАМИ (до 200 символов):
   - Краткое описание контента всего ролика
   - 10-12 релевантных трендовых хештегов для YouTube Shorts
   - Хештеги на русском и английском

Параметры видео:
- Длительность: {duration:.0f} секунд
- Вирусность (AI score): {score}/100
- Ключевые кадры: начало → треть → две трети → конец{speech_context}

Формат ответа (строго):
TITLE: [название без кавычек]
DESCRIPTION: [описание]
HASHTAGS: [#хештег1 #хештег2 ... через пробел]"""

        # Отправляем запрос в Ollama со всеми кадрами
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": "qwen2.5vl:7b",
                "prompt": prompt,
                "images": frames_base64,  # Передаем все 4 кадра
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            },
            timeout=120  # Увеличили timeout для обработки нескольких кадров
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama request failed: {response.text}")

        result = response.json()
        generated_text = result.get("response", "")

        # Парсим ответ
        lines = generated_text.strip().split('\n')
        title = ""
        description = ""
        hashtags = ""

        for line in lines:
            line = line.strip()
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("HASHTAGS:"):
                hashtags = line.replace("HASHTAGS:", "").strip()

        # Если парсинг не удался, используем весь текст как описание
        if not title:
            title = generated_text[:100].strip()

        if not description:
            description = generated_text.strip()

        # Формируем финальное описание с хештегами
        full_description = description
        if hashtags:
            full_description += f"\n\n{hashtags}"

        # Определяем пути для текстовых файлов (убираем .mp4 и добавляем суффикс)
        base_path = output_path.rsplit('.', 1)[0]
        title_file = f"{base_path}_title.txt"
        description_file = f"{base_path}_description.txt"

        # Сохраняем файлы
        with open(title_file, 'w', encoding='utf-8') as f:
            f.write(title)

        with open(description_file, 'w', encoding='utf-8') as f:
            f.write(full_description)

        # Устанавливаем права 644 (rw-r--r--) для чтения из n8n контейнера
        os.chmod(title_file, 0o644)
        os.chmod(description_file, 0o644)

        logger.info(f"Metadata saved: {title_file}, {description_file}")

        return JSONResponse(content={
            'success': True,
            'title': title,
            'description': full_description,
            'title_file': title_file,
            'description_file': description_file,
            'generated_text': generated_text
        })

    except requests.exceptions.Timeout:
        logger.error("Ollama request timeout")
        raise HTTPException(status_code=504, detail="Ollama request timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ollama request error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
