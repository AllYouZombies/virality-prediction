import torch
import logging
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import av
import numpy as np

logger = logging.getLogger(__name__)


class VideoLLaMA3Analyzer:
    """
    Анализирует видео с помощью VideoLLaMA3-2B для определения вирусности.
    Фокусируется на сюжете, динамике, эмоциях и viral-паттернах.
    """

    def __init__(self, model_name: str = "DAMO-NLP-SG/VideoLLaMA3-2B"):
        """
        Args:
            model_name: HuggingFace model ID
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing VideoLLaMA3 ({model_name}) on device: {self.device}")

        try:
            # Загружаем токенизатор и процессор
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Загружаем модель с оптимизацией для GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

            logger.info("VideoLLaMA3 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load VideoLLaMA3 model: {e}")
            raise

    def analyze_virality(
        self,
        video_path: str,
        max_frames: int = 128,
        fps: int = 1
    ) -> Dict[str, Any]:
        """
        Анализирует видео на вирусность с фокусом на TikTok/Shorts паттерны.

        Args:
            video_path: Путь к видео файлу
            max_frames: Максимальное количество кадров для анализа
            fps: Частота извлечения кадров (frames per second)

        Returns:
            Dict с вирусным анализом:
            {
                "virality_score": int (0-100),
                "hook_quality": int (0-100),
                "engagement_potential": int (0-100),
                "viral_elements": List[str],
                "reasoning": str,
                "best_moments": List[Dict],  # таймкоды лучших моментов
                "recommendation": str
            }
        """
        try:
            logger.info(f"Analyzing video virality: {video_path}")

            # Промпт для анализа вирусности TikTok/Shorts
            prompt = """Analyze this video for viral potential on TikTok/YouTube Shorts. Focus on:

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

Be critical and realistic. Only high-quality, truly viral-worthy content should score above 70."""

            # Формируем conversation для модели
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": {
                                "video_path": video_path,
                                "fps": fps,
                                "max_frames": max_frames
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Применяем chat template
            templated_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )

            # Получаем inputs для модели
            inputs = self.processor(
                text=templated_prompt,
                videos=[video_path],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Генерируем ответ
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
                    do_sample=True,
                    temperature=0.3,  # Низкая температура для более точных оценок
                    top_p=0.9
                )

            # Декодируем ответ
            generated_text = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            logger.info(f"VideoLLaMA3 analysis complete: {len(generated_text)} chars")

            # Парсим ответ и извлекаем scores
            analysis = self._parse_analysis(generated_text)

            return analysis

        except Exception as e:
            logger.error(f"VideoLLaMA3 analysis failed: {e}", exc_info=True)
            return {
                "virality_score": 50,
                "hook_quality": 50,
                "engagement_potential": 50,
                "viral_elements": [],
                "reasoning": f"Analysis failed: {str(e)}",
                "best_moments": [],
                "recommendation": "Unable to analyze - using fallback score",
                "raw_response": ""
            }

    def _parse_analysis(self, text: str) -> Dict[str, Any]:
        """
        Парсит текстовый ответ модели и извлекает структурированные данные.
        """
        import re

        # Извлекаем числовые scores
        virality_match = re.search(r'virality score[:\s]+(\d+)', text, re.IGNORECASE)
        hook_match = re.search(r'hook.*?score[:\s]+(\d+)', text, re.IGNORECASE)
        engagement_match = re.search(r'engagement.*?score[:\s]+(\d+)', text, re.IGNORECASE)

        virality_score = int(virality_match.group(1)) if virality_match else 50
        hook_quality = int(hook_match.group(1)) if hook_match else 50
        engagement_potential = int(engagement_match.group(1)) if engagement_match else 50

        # Извлекаем viral elements (если перечислены)
        viral_elements = []
        elements_section = re.search(
            r'viral elements[:\s]+(.*?)(?=\n\n|\nTimestamp|\nReasoning|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if elements_section:
            elements_text = elements_section.group(1)
            viral_elements = [
                line.strip('- •*').strip()
                for line in elements_text.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ][:5]  # Первые 5 элементов

        # Извлекаем reasoning
        reasoning_match = re.search(
            r'reasoning[:\s]+(.*?)(?=\n\n|Recommendation|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else text[:500]

        # Извлекаем recommendation
        recommendation_match = re.search(
            r'recommendation[:\s]+(.*?)(?=\n\n|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        recommendation = recommendation_match.group(1).strip() if recommendation_match else "Publish if score > 70"

        # Ищем timestamps лучших моментов (опционально)
        best_moments = []
        timestamp_pattern = re.finditer(
            r'(\d+:\d+|\d+\.\d+s?)\s*-\s*([^\n]+)',
            text
        )
        for match in timestamp_pattern:
            best_moments.append({
                "timestamp": match.group(1),
                "description": match.group(2).strip()
            })

        return {
            "virality_score": min(100, max(0, virality_score)),
            "hook_quality": min(100, max(0, hook_quality)),
            "engagement_potential": min(100, max(0, engagement_potential)),
            "viral_elements": viral_elements,
            "reasoning": reasoning,
            "best_moments": best_moments[:5],  # Top 5 моментов
            "recommendation": recommendation,
            "raw_response": text
        }

    def get_best_segment_timecodes(
        self,
        video_path: str,
        target_duration: float = 59.0
    ) -> Dict[str, Any]:
        """
        Анализирует видео и предлагает лучший 59-секундный сегмент для вырезки.

        Returns:
            {
                "start_time": float,
                "end_time": float,
                "duration": float,
                "reasoning": str
            }
        """
        try:
            prompt = f"""Watch this video and identify the best {target_duration}-second segment for a viral TikTok/YouTube Short.

Consider:
1. Strong opening hook (first 2-3 seconds of the segment)
2. Clear story arc or payoff within the timeframe
3. High energy and engagement throughout
4. Minimal dead time or boring moments
5. Natural ending point

Provide:
- Start timestamp (MM:SS or SS.SS)
- End timestamp
- Why this segment is the most viral

Be precise with timestamps."""

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": {
                                "video_path": video_path,
                                "fps": 1,
                                "max_frames": 128
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            templated_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=templated_prompt,
                videos=[video_path],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9
                )

            response = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Парсим timestamps
            start_time, end_time = self._parse_timestamps(response)

            return {
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "reasoning": response
            }

        except Exception as e:
            logger.error(f"Segment detection failed: {e}")
            # Fallback: берем середину видео
            return {
                "start_time": 0.0,
                "end_time": target_duration,
                "duration": target_duration,
                "reasoning": f"Fallback to default segment: {str(e)}"
            }

    def _parse_timestamps(self, text: str) -> tuple:
        """Парсит start и end timestamps из текста."""
        import re

        def time_to_seconds(time_str: str) -> float:
            """Конвертирует MM:SS или SS.SS в секунды."""
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return float(time_str.replace('s', ''))

        # Ищем паттерны типа "Start: 1:23" или "Start timestamp: 45.5"
        start_match = re.search(
            r'start.*?[:\s]+(\d+[:\.]\d+|\d+\.\d+|\d+)',
            text,
            re.IGNORECASE
        )
        end_match = re.search(
            r'end.*?[:\s]+(\d+[:\.]\d+|\d+\.\d+|\d+)',
            text,
            re.IGNORECASE
        )

        start_time = time_to_seconds(start_match.group(1)) if start_match else 0.0
        end_time = time_to_seconds(end_match.group(1)) if end_match else 59.0

        return start_time, end_time
