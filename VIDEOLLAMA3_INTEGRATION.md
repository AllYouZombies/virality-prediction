# VideoLLaMA3 Integration

## Overview

VideoLLaMA3-2B от Alibaba DAMO Academy интегрирован для глубокого анализа вирусности видео сегментов.

## Архитектура

### Двухуровневый анализ вирусности:

1. **VideoMAE (быстрая оценка)** - анализирует все окна (3-5 минут каждое)
   - Скорость: ~100 окон/минуту на GPU
   - Результат: базовый score вирусности 0-100

2. **VideoLLaMA3-2B (глубокий анализ)** - анализирует топ-N сегментов
   - Скорость: ~1 сегмент/минуту
   - Результат: детальная оценка с reasoning, hook quality, viral elements

### Workflow

```
Видео 3-5 минут
    ↓
VideoMAE анализ всех окон (быстро)
    ↓
Топ-30 кандидатов по VideoMAE score
    ↓
VideoLLaMA3 глубокий анализ каждого
    ↓
Combined Score = 60% VideoLLaMA3 + 40% VideoMAE
    ↓
Финальный топ-15 для обработки
```

## Модель: VideoLLaMA3-2B

**HuggingFace ID:** `DAMO-NLP-SG/VideoLLaMA3-2B`

### Характеристики:
- **Параметры:** 2 миллиарда
- **Память:** ~4-6GB VRAM
- **Inference:** ~60 секунд на 1 видео (3-5 минут)
- **Входные данные:** 128 кадров @ 1 FPS
- **Возможности:**
  - Понимание сюжета и динамики видео
  - Анализ эмоционального тона
  - Выявление viral-паттернов (hooks, twists, climax)
  - Оценка hook quality (первые 3 секунды)
  - Детектирование трендовых элементов

### Преимущества над VideoMAE:
- ✅ Понимает контекст и сюжет (не только визуальные признаки)
- ✅ Учитывает временную динамику (VideoMAE смотрит только на 16 кадров)
- ✅ Дает reasoning - объясняет, почему видео вирусное
- ✅ State-of-the-art на LVBench и VideoMME бенчмарках

## API Endpoint

### POST `/analyze_timeline_path`

**Изменения в ответе:**

Каждый топовый сегмент теперь содержит:

```json
{
  "start": 120.0,
  "end": 179.0,
  "duration": 59.0,
  "score": 78,  // Combined score (60% VideoLLaMA3 + 40% VideoMAE)
  "is_viral": true,
  "analysis_method": "VideoMAE + VideoLLaMA3",
  "videollama3": {
    "virality_score": 82,
    "hook_quality": 85,
    "engagement_potential": 80,
    "viral_elements": [
      "Strong opening hook",
      "Fast pacing",
      "Emotional payoff",
      "Trending music"
    ],
    "reasoning": "This segment opens with a compelling hook that grabs attention in the first 2 seconds. The pacing is dynamic with quick cuts that maintain engagement. There's a clear emotional arc with a satisfying payoff at the end. The content aligns with current viral trends on TikTok.",
    "recommendation": "Highly recommended for publishing. Consider trimming the intro by 1 second for maximum impact."
  }
}
```

## Производительность

### Время обработки (типичное видео 5 минут):

| Этап | Время | GPU Memory |
|------|-------|------------|
| VideoMAE (100 окон) | ~1 минута | 2GB |
| VideoLLaMA3 (30 топов) | ~30 минут | +6GB |
| **Итого** | **~31 минута** | **8GB** |

### Оптимизация:

При необходимости можно:
- Уменьшить количество глубоко анализируемых сегментов (30 → 15)
- Снизить max_frames в VideoLLaMA3 (128 → 64)
- Использовать квантизацию модели (int8)

## Использование VRAM

На GPU с 12GB VRAM одновременно работают:

| Модель | VRAM | Статус |
|--------|------|--------|
| VideoMAE | ~2GB | Загружена всегда |
| Silero VAD | ~0.5GB | Загружена всегда |
| Whisper Base | ~1.5GB | Загружена всегда |
| VideoLLaMA3-2B | ~6GB | Ленивая загрузка при первом использовании |
| **Свободно** | **~2GB** | Резерв для обработки |

## Настройка

### Изменение веса моделей в Combined Score:

В `/virality-prediction/app/main.py` строка 543-547:

```python
# Объединяем scores: 60% VideoLLaMA3 + 40% VideoMAE
combined_score = int(
    llama_analysis['virality_score'] * 0.6 +
    segment['score'] * 0.4
)
```

Можно изменить на:
- `0.7 / 0.3` - больше доверия VideoLLaMA3
- `0.5 / 0.5` - равные веса
- `0.8 / 0.2` - почти полное доверие VideoLLaMA3

### Изменение количества глубоко анализируемых сегментов:

Строка 503:

```python
preliminary_top = sorted(results, key=lambda x: x['score'], reverse=True)[:top_n * 2]
```

`top_n * 2` означает: если `top_n=15`, анализируем 30 кандидатов.

Можно изменить на:
- `top_n * 1` - анализируем ровно top_n (быстрее, но может пропустить хорошие)
- `top_n * 3` - анализируем больше (медленнее, но точнее)

## Prompt для VideoLLaMA3

Промпт оптимизирован для TikTok/YouTube Shorts контента:

```python
"""Analyze this video for viral potential on TikTok/YouTube Shorts. Focus on:

1. HOOK (first 3 seconds): How compelling is the opening?
2. PACING: Is the content dynamic and fast-paced?
3. EMOTIONAL IMPACT: Does it evoke strong emotions (surprise, joy, shock)?
4. VISUAL APPEAL: Are the visuals eye-catching and high-quality?
5. STORYTELLING: Is there a clear narrative arc or payoff?
6. RELATABILITY: Will viewers find it relatable or shareable?
7. TREND ALIGNMENT: Does it align with current viral trends?

Provide:
- Overall virality score (0-100)
- Hook quality score (0-100)
- Engagement potential (0-100)
- List of viral elements present
- Specific timestamps of the most viral moments
- Reasoning for the scores
- Recommendation: should we publish this?

Be critical and realistic. Only high-quality, truly viral-worthy content should score above 70.
"""
```

## Мониторинг

### Логи VideoLLaMA3:

```bash
docker compose logs virality-api -f | grep VideoLLaMA3
```

### Проверка загрузки модели:

```bash
curl http://localhost:8000/health
```

В ответе должно быть `"videollama3_loaded": true` после первого использования.

## Troubleshooting

### Ошибка: Out of Memory (OOM)

**Решение:**
1. Перезапустить контейнер: `docker compose restart virality-api`
2. Уменьшить max_frames: в `videollama3_analyzer.py` строка 534 изменить `max_frames=128` → `max_frames=64`

### VideoLLaMA3 analysis failed

Проверить логи:
```bash
docker compose logs virality-api | grep "VideoLLaMA3 analysis failed"
```

Типичные причины:
- Недостаточно VRAM → уменьшить max_frames
- Модель не скачалась → проверить HuggingFace доступность
- Ошибка ffmpeg при вырезке сегмента → проверить rights на /tmp

### Fallback на VideoMAE only

Если VideoLLaMA3 не может загрузиться, система автоматически продолжит работу только с VideoMAE:

```python
segment['analysis_method'] = 'VideoMAE only'
```

## Дальнейшие улучшения

### Потенциальные апгрейды:

1. **VideoLLaMA3-7B** вместо 2B
   - +50% точность
   - +6GB VRAM (требуется 18GB GPU)
   - Время анализа: +30%

2. **Кэширование анализа**
   - Сохранять VideoLLaMA3 результаты в Redis
   - Не переанализировать идентичные сегменты

3. **Batch inference VideoLLaMA3**
   - Анализировать несколько сегментов параллельно
   - Требует модификацию модели

4. **Fine-tuning на TikTok данных**
   - Собрать датасет вирусных/невирусных TikTok видео
   - Дообучить VideoLLaMA3-2B
   - Увеличить точность на +20-30%

## Credits

- **VideoLLaMA3:** Alibaba DAMO Academy (https://github.com/DAMO-NLP-SG/VideoLLaMA3)
- **Paper:** https://arxiv.org/abs/2501.13106
- **HuggingFace:** https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-2B
