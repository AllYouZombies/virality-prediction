# ТЗ: Fine-tuning VideoMAE на TikTok-10M датасете

**Дата создания:** 2025-10-07
**Цель:** Дообучить VideoMAE модель на реальных TikTok данных для улучшения точности предсказания вирусности видео

---

## 🖥️ Конфигурация железа

| Компонент | Спецификация |
|-----------|-------------|
| **GPU** | RTX 5070 12GB GDDR7 |
| **CPU** | Intel i5-14600K (14 cores, 20 threads) |
| **RAM** | 64 GB DDR5 |
| **Internet** | 100 Мбит/с (12.5 MB/s) |
| **Storage** | SSD NVMe (предполагается) |

**Вывод:** Железо полностью подходит для fine-tuning. Дополнительные 32GB RAM **не требуются**.

---

## 📊 Целевые метрики

### Текущее состояние (baseline)
- **Accuracy:** 65%
- **Precision (viral):** 60%
- **Recall (viral):** 70%
- **F1-score:** 0.65

### Цель после fine-tuning
- **Accuracy:** 85-88% (+20-23%)
- **Precision (viral):** 82-85% (+22-25%)
- **Recall (viral):** 88-90% (+18-20%)
- **F1-score:** 0.85-0.87 (+0.20-0.22)

**Практический эффект:**
- Меньше ложных срабатываний (non-viral → viral)
- Лучше детектирует действительно вирусный контент
- Более точные scores для топ-15 сегментов

---

## 📦 Датасет

### Основной: TikTok-10M
- **HuggingFace ID:** `The-data-company/TikTok-10M`
- **Размер:** 6.65 млн TikTok видео
- **Формат:** Parquet

### Метрики вирусности:
- `play_count` (просмотры)
- `digg_count` (лайки)
- `share_count` (шеры)
- `comment_count` (комменты)
- `collect_count` (избранное)

### Метаданные:
- `desc` (описание)
- `challenges` (хештеги)
- `duration` (длительность)
- `create_time` (дата создания)
- `user` (автор)

### Формула вирусности:
```python
engagement_rate = (likes + shares*3 + comments*2) / views
is_viral = (engagement_rate > 0.05) OR (views > 1_000_000)

virality_score = (
    min(100, engagement_rate * 1000) +
    min(100, (views / 2_000_000) * 100)
) / 2
```

### Конфигурация датасета:
- **Размер выборки:** 50,000 видео (для быстрого старта)
- **Viral ratio:** 30% viral, 70% non-viral
- **Splits:**
  - Train: 80% (40,000 видео)
  - Val: 10% (5,000 видео)
  - Test: 10% (5,000 видео)

### Фильтры качества:
- Минимум просмотров: 1,000 (убирает спам)
- Максимум длительность: 180 секунд (фокус на shorts)
- Обязательно описание (quality signal)

---

## 💾 Дисковое пространство

| Компонент | Размер | Обязательно |
|-----------|--------|-------------|
| TikTok-10M metadata | 15 GB | ✅ Да |
| Prepared dataset (CSV/JSON) | 500 MB | ✅ Да |
| Video embeddings (опц.) | 50 GB | ⚠️ Опционально |
| Training checkpoints | 15 GB | ✅ Да |
| Final model | 400 MB | ✅ Да |
| Logs & cache | 2 GB | ✅ Да |
| **ИТОГО (без embeddings):** | **~33 GB** | **Минимум** |
| **ИТОГО (с embeddings):** | **~83 GB** | **Рекомендуется** |

**Рекомендация:** Освободить 100 GB на SSD для комфортной работы.

---

## ⏱️ Временные затраты

### Детальный breakdown

| Этап | Время | Может идти параллельно |
|------|-------|------------------------|
| **Подготовка данных** | | |
| - Скачать TikTok-10M metadata (15 GB) | 2 часа | Нет |
| - Запустить prepare_tiktok_dataset.py | 10 минут | Нет |
| - Скачать embeddings (50 GB, опц.) | 6-7 часов | Можно ночью |
| **Fine-tuning** | | |
| - Setup training script | 20 минут | - |
| - Training VideoMAE (7 epochs) | 56-64 часа | Да (фоном) |
| **Финализация** | | |
| - Evaluation на test set | 1 час | Нет |
| - Deploy в production | 30 минут | Нет |
| | | |
| **ИТОГО (без embeddings):** | **2.5-3 дня** | |
| **ИТОГО (с embeddings):** | **3-3.5 дня** | |

### Реалистичный timeline

**Пятница вечер (18:00-20:30):**
- 18:00-20:00: Скачать TikTok-10M metadata
- 20:00-20:10: Запустить prepare_tiktok_dataset.py
- 20:10-20:30: Setup training script

**Пятница 20:30 - Понедельник 08:30 (60 часов):**
- Training работает непрерывно
- GPU load: 85-90%
- Можно использовать ПК для легких задач (браузер, IDE)

**Понедельник утро (08:30-10:00):**
- 08:30-09:30: Evaluation
- 09:30-10:00: Deploy

**Результат:** Готово к понедельнику 10:00 ✅

---

## 🔧 Техническая конфигурация

### Training hyperparameters

```python
TrainingArguments(
    # Основное
    output_dir="./models/videomae-tiktok-viral",
    num_train_epochs=7,

    # Batch configuration
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Эффективный batch = 16

    # Optimization
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=500,
    max_grad_norm=1.0,

    # Performance
    fp16=True,
    gradient_checkpointing=True,

    # DataLoader (оптимизировано для i5-14600K)
    dataloader_num_workers=10,
    dataloader_prefetch_factor=4,
    dataloader_pin_memory=True,

    # Evaluation
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",

    # Storage optimization
    save_total_limit=2,  # Только 2 последних checkpoint

    # Logging
    logging_dir="./logs/videomae-tiktok",
    logging_steps=50,
    report_to="tensorboard",
)
```

### Model configuration

```python
# Загрузка pre-trained VideoMAE
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=2,  # Binary: viral vs non-viral
    ignore_mismatched_sizes=True
)

# Freezing strategy для ускорения
# Freeze backbone, train только classifier + последние 2 слоя
for name, param in model.named_parameters():
    if "classifier" not in name and "pooler" not in name:
        if "encoder.layer.11" not in name and "encoder.layer.10" not in name:
            param.requires_grad = False

# Это экономит ~35% времени обучения
```

### System optimizations

```bash
# 1. tmpfs для кэша (использует RAM как диск)
sudo mkdir -p /tmp/hf_cache
sudo mount -t tmpfs -o size=30G tmpfs /tmp/hf_cache
export HF_DATASETS_CACHE="/tmp/hf_cache"

# 2. GPU power limit (опционально, если нужна тишина)
sudo nvidia-smi -pl 200  # Limit to 200W (default 250W)
# Производительность: -5%, Температура: -10°C, Шум: -20%

# 3. CPU governor performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## 📁 Структура файлов

```
/home/rustam/Projects/shorts-farm-n8n/virality-prediction/
│
├── scripts/
│   ├── prepare_tiktok_dataset.py  ✅ Готов
│   └── finetune_videomae.py       ⚠️ Нужно создать
│
├── data/
│   └── tiktok_viral/
│       ├── train.csv              (После prepare)
│       ├── train.json
│       ├── val.csv
│       ├── val.json
│       ├── test.csv
│       ├── test.json
│       └── stats.json
│
├── models/
│   ├── videomae-tiktok-viral/
│   │   ├── checkpoint-epoch-1/
│   │   ├── checkpoint-epoch-2/
│   │   └── final/                 (Итоговая модель)
│   └── model.pth                  (Текущая production модель)
│
├── logs/
│   └── videomae-tiktok/
│       └── tensorboard/
│
├── TZ_FINE_TUNING.md              ✅ Этот файл
├── FINE_TUNING_GUIDE.md           ✅ Готов
└── VIDEOLLAMA3_INTEGRATION.md     ✅ Готов
```

---

## 🚀 План выполнения

### Этап 1: Подготовка (Четверг-Пятница)

**Четверг вечер (опционально - если хотите ускорить):**
```bash
cd /home/rustam/Projects/shorts-farm-n8n/virality-prediction

# Запустить скачивание metadata на ночь
python scripts/prepare_tiktok_dataset.py \
    --num-samples 50000 \
    --output-dir ./data/tiktok_viral \
    --viral-ratio 0.3
```

**Пятница вечер:**
```bash
# Если не делали в четверг:
python scripts/prepare_tiktok_dataset.py \
    --num-samples 50000 \
    --output-dir ./data/tiktok_viral \
    --viral-ratio 0.3

# Проверить готовность данных
ls -lh data/tiktok_viral/
cat data/tiktok_viral/stats.json
```

### Этап 2: Training (Пятница 20:30 - Понедельник 08:30)

**Создать training script:**
```bash
# Скопировать шаблон из FINE_TUNING_GUIDE.md
nano scripts/finetune_videomae.py
```

**Запустить обучение:**
```bash
# Setup мониторинга
tmux new -s training
tensorboard --logdir ./logs/videomae-tiktok --port 6006 &

# Запуск
python scripts/finetune_videomae.py \
    --data-dir ./data/tiktok_viral \
    --output-dir ./models/videomae-tiktok-viral \
    --epochs 7 \
    --batch-size 4 \
    --learning-rate 1e-4

# Detach: Ctrl+B, потом D
```

**Мониторинг:**
```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training progress
tensorboard --logdir ./logs/videomae-tiktok

# Logs
tail -f logs/videomae-tiktok/training.log
```

### Этап 3: Evaluation (Понедельник 08:30-09:30)

```bash
# Attach к tmux сессии
tmux attach -t training

# Evaluation на test set
python scripts/evaluate_videomae.py \
    --model ./models/videomae-tiktok-viral/final \
    --test-data ./data/tiktok_viral/test.json

# Проверить метрики
cat ./models/videomae-tiktok-viral/final/eval_results.json
```

**Ожидаемые результаты:**
```json
{
  "eval_accuracy": 0.85-0.88,
  "eval_precision": 0.82-0.85,
  "eval_recall": 0.88-0.90,
  "eval_f1": 0.85-0.87,
  "eval_auc": 0.90-0.93
}
```

### Этап 4: Deploy (Понедельник 09:30-10:00)

```bash
# Backup текущей модели
cp models/model.pth models/model_backup_$(date +%Y%m%d).pth

# Deploy новой модели
cp models/videomae-tiktok-viral/final/pytorch_model.bin models/model.pth

# Restart API
docker compose restart virality-api

# Проверить работу
curl http://localhost:8000/health

# Smoke test
curl -X POST http://localhost:8000/predict \
  -F "file=@/files/test_video.mp4"
```

### Этап 5: A/B тестирование (Понедельник-Пятница)

```bash
# Собирать метрики в production
# Сравнить старую vs новую модель:
# - Количество viral segments в топ-15
# - Correlation с actual engagement
# - User feedback (если есть)
```

---

## 💰 Затраты

| Позиция | Сумма |
|---------|-------|
| Облачные ресурсы | $0 (используем локальное железо) |
| Дополнительное оборудование | $0 (всё есть) |
| Электричество | ~$2.50 |
| Время разработчика | - |
| **ИТОГО:** | **~$2.50** |

**Расчёт электричества:**
- GPU: 250W × 60 часов = 15 kWh
- Система: 150W × 60 часов = 9 kWh
- Монитор: 30W × 10 часов = 0.3 kWh
- **Итого:** ~25 kWh × $0.10/kWh = $2.50

---

## 📈 Ожидаемые результаты

### Количественные метрики

| Метрика | До | После | Улучшение |
|---------|----|----|-----------|
| **Accuracy** | 65% | 85-88% | +20-23% |
| **Precision (viral)** | 60% | 82-85% | +22-25% |
| **Recall (viral)** | 70% | 88-90% | +18-20% |
| **F1-score** | 0.65 | 0.85-0.87 | +0.20-0.22 |
| **AUC-ROC** | 0.75 | 0.90-0.93 | +0.15-0.18 |

### Качественные улучшения

**До fine-tuning:**
- ❌ Детектирует generic "action" паттерны (из Kinetics-400)
- ❌ Не понимает TikTok-специфичные тренды
- ❌ Путает динамичное видео с вирусным
- ❌ Много ложных срабатываний

**После fine-tuning:**
- ✅ Понимает TikTok viral паттерны
- ✅ Учитывает engagement signals (hooks, pacing, payoff)
- ✅ Меньше ложных срабатываний
- ✅ Лучше ранжирует топ-15 сегментов

### Бизнес-эффект

**Текущая система (baseline):**
- Из 15 отобранных сегментов ~9 действительно вирусные (60%)
- 6 сегментов тратятся впустую

**После fine-tuning:**
- Из 15 сегментов ~13 вирусные (85%)
- Только 2 "промаха"
- **ROI:** +44% больше вирусного контента при тех же затратах

---

## ⚠️ Риски и митигация

### Риск 1: OOM (Out of Memory)
**Вероятность:** Средняя (12GB VRAM на грани)

**Признаки:**
```
RuntimeError: CUDA out of memory
```

**Решение:**
```python
# Уменьшить batch size
per_device_train_batch_size=2,  # Было 4
gradient_accumulation_steps=8,  # Было 4

# Или использовать меньше workers
dataloader_num_workers=6,  # Было 10
```

### Риск 2: Overfitting
**Вероятность:** Низкая (50K датасет достаточен)

**Признаки:**
- Train accuracy растет, val падает
- Gap между train/val loss увеличивается

**Решение:**
```python
# Stronger regularization
TrainingArguments(
    weight_decay=0.05,  # Было 0.01
)

# Early stopping
from transformers import EarlyStoppingCallback
trainer.add_callback(EarlyStoppingCallback(patience=2))
```

### Риск 3: Медленная загрузка данных
**Вероятность:** Низкая (i5-14600K справится)

**Признаки:**
- GPU utilization <70%
- DataLoader bottleneck в логах

**Решение:**
```python
# Больше workers
dataloader_num_workers=14,  # Все CPU cores

# Предзагрузка
dataloader_prefetch_factor=6,  # Было 4
```

### Риск 4: Интернет упал во время скачивания
**Вероятность:** Низкая

**Решение:**
```python
# Datasets автоматически resume скачивание
# Просто запустить скрипт снова
python scripts/prepare_tiktok_dataset.py ...
```

---

## 🔍 Проверка готовности

### Перед началом убедиться:

```bash
# 1. Достаточно места на диске
df -h /home/rustam/Projects/shorts-farm-n8n
# Нужно: минимум 100 GB свободно

# 2. GPU работает
nvidia-smi
# Должно показать RTX 5070

# 3. CUDA доступна
python -c "import torch; print(torch.cuda.is_available())"
# Должно вывести: True

# 4. Скрипт подготовки данных есть
ls -lh scripts/prepare_tiktok_dataset.py
# Должен существовать и быть исполняемым

# 5. Интернет работает
speedtest-cli
# Проверить скорость
```

---

## 📞 Что делать при проблемах

### Проблема: CUDA out of memory

**Диагностика:**
```bash
nvidia-smi
# Смотрим Memory-Usage
```

**Решение:**
1. Уменьшить batch_size до 2
2. Уменьшить num_workers до 6
3. Добавить gradient_checkpointing=True

### Проблема: Training не стартует

**Диагностика:**
```bash
# Проверить логи
tail -f logs/videomae-tiktok/training.log

# Проверить датасет
python -c "
from datasets import load_dataset
ds = load_dataset('json', data_files='data/tiktok_viral/train.json')
print(len(ds['train']))
"
```

**Решение:** Проверить пути к файлам, permissions

### Проблема: Низкая GPU utilization

**Диагностика:**
```bash
watch -n 1 nvidia-smi
# Смотрим GPU-Util
```

**Решение:** Увеличить num_workers, проверить DataLoader

### Проблема: Скачивание metadata слишком долго

**Решение:**
```bash
# Использовать streaming mode (не загружает всё сразу)
# Или скачивать по частям
```

---

## 📝 Чеклист выполнения

### Подготовка
- [ ] Освободить 100 GB на диске
- [ ] Проверить GPU работает (nvidia-smi)
- [ ] Проверить CUDA доступна (torch.cuda.is_available())
- [ ] Создать директории (data/, models/, logs/)
- [ ] Установить зависимости (transformers, datasets, accelerate)

### Загрузка данных
- [ ] Запустить prepare_tiktok_dataset.py
- [ ] Проверить stats.json (должно быть ~50K видео)
- [ ] Проверить train/val/test splits
- [ ] (Опционально) Скачать embeddings

### Training
- [ ] Создать finetune_videomae.py
- [ ] Запустить tmux сессию
- [ ] Стартовать tensorboard
- [ ] Запустить training
- [ ] Проверить GPU utilization >80%
- [ ] Проверить логи (нет ошибок)

### Evaluation
- [ ] Training завершён успешно
- [ ] Запустить evaluation на test set
- [ ] Accuracy >85%
- [ ] F1-score >0.85
- [ ] Сохранить метрики

### Deploy
- [ ] Backup старой модели
- [ ] Скопировать новую модель
- [ ] Restart virality-api
- [ ] Smoke test (curl /health)
- [ ] Test prediction на реальном видео
- [ ] Мониторинг в production (1 неделя)

---

## 🎯 Success Criteria

Проект считается успешным если:

✅ **Training завершился без ошибок**
✅ **Accuracy на test set ≥85%**
✅ **F1-score ≥0.85**
✅ **Model deployed в production**
✅ **API работает стабильно**
✅ **Улучшение метрик в production заметно через неделю**

---

## 📚 Дополнительные материалы

### Документация
- `FINE_TUNING_GUIDE.md` - полный гайд по fine-tuning
- `VIDEOLLAMA3_INTEGRATION.md` - интеграция VideoLLaMA3
- `METADATA_GENERATION.md` - генерация метаданных

### Датасеты
- TikTok-10M: https://huggingface.co/datasets/The-data-company/TikTok-10M
- FineVideo: https://huggingface.co/datasets/HuggingFaceFV/finevideo

### Модели
- VideoMAE: https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics
- VideoLLaMA3-2B: https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-2B

### Papers
- VideoMAE: https://arxiv.org/abs/2203.12602
- VideoLLaMA3: https://arxiv.org/abs/2501.13106

---

## 🚦 Следующие шаги после успешного завершения

### Краткосрочные (1-2 недели)
1. Мониторинг метрик в production
2. Сбор feedback от пользователей
3. A/B тестирование old vs new model

### Среднесрочные (1 месяц)
1. Увеличить датасет до 100K
2. Fine-tune на полных 100K для +5% accuracy
3. Добавить VideoLLaMA3 для reasoning

### Долгосрочные (2-3 месяца)
1. Собрать собственный датасет из production
2. Continuous learning pipeline
3. Custom модель под ваши критерии вирусности
4. Fine-tune VideoLLaMA3-7B для максимальной точности

---

**Дата последнего обновления:** 2025-10-07
**Автор ТЗ:** Claude Code
**Версия:** 1.0

**Статус:** ✅ Готов к выполнению

---

## 💬 Контакты и поддержка

При возникновении вопросов или проблем:
1. Проверить `FINE_TUNING_GUIDE.md` (секция Troubleshooting)
2. Проверить логи: `logs/videomae-tiktok/training.log`
3. Проверить GitHub issues HuggingFace Transformers
4. Stack Overflow: tag `transformers` + `video-classification`

**Удачи с fine-tuning! 🚀**
