# Fine-Tuning Guide: Обучение моделей на TikTok данных

## 🎯 Цель

Дообучить VideoMAE и VideoLLaMA3 на реальных TikTok данных для улучшения точности предсказания вирусности.

## 📊 Рекомендованные датасеты

### 1. TikTok-10M ⭐⭐⭐⭐⭐ (Основной)

**HuggingFace:** `The-data-company/TikTok-10M`

**Содержит:**
- 6.65 млн TikTok видео
- Метрики: views, likes, shares, comments, favorites
- Метаданные: описание, хештеги, длительность, дата

**Почему идеален:**
- ✅ Реальные метрики вирусности
- ✅ Разнообразный контент (все категории TikTok)
- ✅ Большой размер (достаточно для fine-tuning)
- ✅ Готовые labels можно вычислить из engagement метрик

### 2. FineVideo (Дополнительный)

**HuggingFace:** `HuggingFaceFV/finevideo`

**Содержит:**
- 43,751 YouTube видео (3,425 часов)
- Аннотации: mood, storytelling, scenes, QA
- 122 категории контента

**Применение:**
- Дообучение VideoLLaMA3 на understanding сюжета и эмоций
- Не содержит метрик вирусности, но учит анализировать content

## 🚀 Быстрый старт

### Шаг 1: Подготовка датасета

```bash
cd /home/rustam/Projects/shorts-farm-n8n/virality-prediction

# Создать директорию для скриптов
mkdir -p scripts
chmod +x scripts/prepare_tiktok_dataset.py

# Запустить подготовку датасета (100K видео)
python scripts/prepare_tiktok_dataset.py \
    --output-dir ./data/tiktok_viral \
    --num-samples 100000 \
    --viral-ratio 0.3
```

**Что делает скрипт:**
1. Загружает TikTok-10M с HuggingFace
2. Фильтрует видео:
   - Минимум 1K views (убирает спам)
   - Максимум 180 секунд (фокус на shorts)
   - Есть описание (quality signal)
3. Вычисляет virality на основе формулы:
   ```
   engagement_rate = (likes + shares*3 + comments*2) / views
   is_viral = (engagement_rate > 5%) OR (views > 1M)
   ```
4. Балансирует датасет (30% viral, 70% non-viral)
5. Делит на train/val/test (80%/10%/10%)
6. Сохраняет в CSV и JSON

**Результат:**
```
data/tiktok_viral/
├── train.csv       # 80K видео с labels
├── train.json
├── val.csv         # 10K видео
├── val.json
├── test.csv        # 10K видeo
├── test.json
└── stats.json      # Статистика датасета
```

### Шаг 2: Загрузка видео (опционально)

TikTok-10M содержит только метаданные, не сами видео. Для fine-tuning нужно:

**Вариант А: Использовать video embeddings (рекомендуется)**
```python
# Используем pre-extracted features из TikTok-10M
# Многие видео уже имеют embeddings
```

**Вариант Б: Скачать видео по ID**
```python
# Использовать TikTok API или scraper
# (требует API ключ или прокси)
```

**Вариант В: Использовать синтетические данные**
```python
# Генерировать похожие видео с помощью Stable Video Diffusion
# Или augment существующие видео из UCF-101/Kinetics
```

### Шаг 3: Fine-tuning VideoMAE

Создайте `scripts/finetune_videomae.py`:

```python
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Загрузка prepared dataset
dataset = load_dataset("json", data_files={
    "train": "data/tiktok_viral/train.json",
    "val": "data/tiktok_viral/val.json",
    "test": "data/tiktok_viral/test.json"
})

# Загрузка pre-trained VideoMAE
model = VideoMAEForVideoClassification.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics",
    num_labels=2,  # Binary: viral vs non-viral
    ignore_mismatched_sizes=True  # Заменяем classifier head
)

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/videomae-tiktok-viral",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # Mixed precision для экономии памяти
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=500,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,  # Define your metrics
)

# Train!
trainer.train()

# Evaluate
results = trainer.evaluate(dataset["test"])
print(f"Test Accuracy: {results['eval_accuracy']:.2%}")

# Save model
model.save_pretrained("./models/videomae-tiktok-viral-final")
```

**Запуск:**
```bash
python scripts/finetune_videomae.py
```

**Ожидаемое время:**
- На GPU 12GB: ~5-7 дней
- На GPU 24GB: ~2-3 дня

**Ожидаемые результаты:**
- Baseline (без fine-tuning): ~65% accuracy
- После fine-tuning: ~85-90% accuracy

### Шаг 4: Fine-tuning VideoLLaMA3 (Advanced)

**Требования:**
- GPU 24GB+ (или multi-GPU)
- Parameter-Efficient Fine-Tuning (LoRA/QLoRA)

```python
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer
from peft import LoraConfig, get_peft_model

# Загрузка VideoLLaMA3-2B
model = AutoModelForCausalLM.from_pretrained(
    "DAMO-NLP-SG/VideoLLaMA3-2B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# LoRA config (экономит память)
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Применяем LoRA
model = get_peft_model(model, lora_config)

# Training с instruction tuning
# Промпты формата:
# "Analyze this TikTok video with {views} views and {engagement_rate}% engagement.
#  Explain why it is {viral/non-viral}."
```

## 📈 Метрики для оценки

### Для VideoMAE (Classification):

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    auc = roc_auc_score(labels, pred.predictions[:, 1])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
```

**Целевые метрики:**
- Accuracy: >85%
- F1-score: >0.80
- AUC-ROC: >0.90

### Для VideoLLaMA3 (Generation):

- BLEU/ROUGE для quality reasoning
- Human evaluation для usefulness
- Correlation с actual virality (Pearson's r)

## 🎛️ Гиперпараметры

### VideoMAE Fine-tuning:

```python
# Рекомендованные настройки для 12GB GPU
TrainingArguments(
    num_train_epochs=10,              # 10-20 epochs достаточно
    per_device_train_batch_size=4,    # 4-8 в зависимости от VRAM
    learning_rate=1e-4,                # 1e-4 до 1e-5
    weight_decay=0.01,                 # L2 regularization
    warmup_steps=500,                  # Gradual lr warmup
    fp16=True,                         # Mixed precision
    gradient_accumulation_steps=4,     # Effective batch size = 16
    max_grad_norm=1.0,                 # Gradient clipping
)
```

**Freezing strategy:**
```python
# Freeze backbone, train только classifier
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False
```

### VideoLLaMA3 LoRA:

```python
LoraConfig(
    r=16,                # LoRA rank (8-32)
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.1,    # Dropout для regularization
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

## 💡 Tips для улучшения результатов

### 1. Data Augmentation

```python
# Для видео:
- Random crop
- Color jitter
- Temporal crop (разные временные отрезки)
- Speed augmentation (0.8x - 1.2x)
```

### 2. Class Balancing

```python
# Используйте weighted loss для несбалансированных классов
class_weights = torch.tensor([1.0, 2.0])  # Больше вес для viral класса
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 3. Transfer Learning Pipeline

```
UCF-101 (action recognition)
    ↓
Kinetics-400 (pre-training)
    ↓
TikTok-10M (fine-tuning)
    ↓
Your specific domain
```

### 4. Ensemble Models

```python
# Combine predictions from multiple models
final_score = (
    videomae_score * 0.4 +
    videollama3_score * 0.6
)
```

## 🐛 Troubleshooting

### OOM (Out of Memory)

**Решения:**
- Уменьшить batch size
- Включить gradient checkpointing
- Использовать gradient accumulation
- Freeze больше layers

```python
# Gradient checkpointing (экономит память)
model.gradient_checkpointing_enable()
```

### Overfitting

**Признаки:**
- Train accuracy >> Val accuracy
- Loss на train падает, на val растёт

**Решения:**
- Увеличить dropout (0.1 → 0.3)
- Stronger data augmentation
- Early stopping
- Меньше epochs

### Underfitting

**Признаки:**
- Low accuracy на train и val
- Loss не падает

**Решения:**
- Больше epochs
- Выше learning rate
- Unfreeze больше layers
- Bigger model (2B → 7B)

## 📚 Ресурсы

### Papers:
- VideoMAE: https://arxiv.org/abs/2203.12602
- VideoLLaMA3: https://arxiv.org/abs/2501.13106
- LoRA: https://arxiv.org/abs/2106.09685

### Datasets:
- TikTok-10M: https://huggingface.co/datasets/The-data-company/TikTok-10M
- FineVideo: https://huggingface.co/datasets/HuggingFaceFV/finevideo
- UCF-101: https://huggingface.co/datasets/flwrlabs/ucf101

### Models:
- VideoMAE: https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics
- VideoLLaMA3-2B: https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-2B

## 🚦 Roadmap

### Phase 1: Quick Win (1-2 недели)
- ✅ Подготовить TikTok-10M датасет
- ✅ Fine-tune VideoMAE на 100K видео
- ✅ Evaluate и deploy

**Ожидаемое улучшение:** +15-20% accuracy

### Phase 2: Advanced (1 месяц)
- Fine-tune VideoLLaMA3 с LoRA
- Добавить FineVideo для reasoning
- Ensemble VideoMAE + VideoLLaMA3

**Ожидаемое улучшение:** +25-30% accuracy

### Phase 3: Production (2-3 месяца)
- Collect собственные viral TikTok данные
- Continuous learning pipeline
- A/B тестирование моделей
- Monitor drift и retrain

**Ожидаемое улучшение:** +35-40% accuracy

## ⚡ Итоговая рекомендация

**Начните с TikTok-10M + VideoMAE:**
1. Самый быстрый путь к улучшению
2. Требует только 12GB GPU
3. Можно обучить за неделю
4. Immediate impact на точность

**Затем добавьте VideoLLaMA3:**
- Для более глубокого анализа
- Объяснение reasoning
- Детектирование трендов

**В будущем:**
- Соберите собственный датасет из вашего production traffic
- Continuous learning на новых данных
- Custom модель специально под ваши критерии вирусности
