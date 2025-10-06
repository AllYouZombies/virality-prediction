import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TrimmingOptimizerV2:
    """
    Простой алгоритм удаления низковирусных фрагментов на основе шагов.

    Алгоритм:
    1. Делим окно на шаги (step_size = min_remove_duration)
    2. Находим N худших шагов для удаления
    3. Группируем соседние шаги
    4. Корректируем границы по паузам речи (±step_size)
    """

    def __init__(
        self,
        target_duration: float = 60.0,
        step_size: float = 10.0,
        speech_gap_tolerance: float = 0.3  # 300 мс - допустимая пауза для обрезки
    ):
        self.target_duration = target_duration
        self.max_duration = 59.0  # Максимальная длительность ролика
        self.min_duration = 50.0  # Минимальная длительность ролика
        self.step_size = step_size
        self.speech_gap_tolerance = speech_gap_tolerance
        logger.info(f"TrimmingOptimizerV2: target={target_duration}s, max={self.max_duration}s, "
                   f"min={self.min_duration}s, step={step_size}s, speech_gap={speech_gap_tolerance}s")

    def optimize_segment(
        self,
        micro_segments: List[Dict],
        speech_segments: List[Dict[str, float]],
        window_start: float,
        window_duration: float,
        is_last_segment: bool
    ) -> Dict:
        """
        Оптимизирует сегмент видео до целевой длительности.

        Returns:
            {
                'keep_ranges': [(start, end), ...],
                'remove_ranges': [(start, end), ...],
                'final_duration': float,
                'removed_duration': float,
                'avg_score': float
            }
        """
        # Проверяем, нужно ли обрезать видео
        if window_duration <= self.max_duration:
            # Окно уже в допустимом диапазоне (≤59 секунд)
            return {
                'keep_ranges': [(window_start, window_start + window_duration)],
                'remove_ranges': [],
                'final_duration': window_duration,
                'removed_duration': 0,
                'avg_score': np.mean([s['score'] for s in micro_segments])
            }

        # Вычисляем, сколько нужно удалить для достижения max_duration (59 сек)
        to_remove = window_duration - self.max_duration

        # Шаг 1: Группируем микро-сегменты в шаги
        steps = self._create_steps(micro_segments, window_start, window_duration)
        logger.info(f"Created {len(steps)} steps from {len(micro_segments)} micro-segments")

        # Шаг 2: Находим худшие шаги для удаления
        steps_to_remove_count = int(np.ceil(to_remove / self.step_size))
        worst_steps = self._find_worst_steps(steps, steps_to_remove_count)
        logger.info(f"Need to remove {steps_to_remove_count} steps (~{to_remove:.1f}s)")

        # Шаг 3: Группируем соседние шаги
        step_groups = self._group_consecutive_steps(worst_steps)
        logger.info(f"Grouped into {len(step_groups)} removal groups")

        # Шаг 4: Корректируем границы по паузам речи
        remove_ranges = []
        window_end = window_start + window_duration
        for group in step_groups:
            adjusted_range = self._adjust_boundaries_for_speech(
                group, speech_segments, window_start, window_end
            )
            if adjusted_range:
                remove_ranges.append(adjusted_range)

        # Вычисляем keep_ranges
        keep_ranges = self._compute_keep_ranges(window_start, window_duration, remove_ranges)

        final_duration = sum(e - s for s, e in keep_ranges)
        removed_duration = sum(e - s for s, e in remove_ranges)

        # Проверяем, что итоговая длина не превышает max_duration
        # Если превышает, удаляем еще один худший шаг
        iteration = 0
        while final_duration > self.max_duration and iteration < 10:
            logger.warning(f"Final duration {final_duration:.1f}s exceeds max {self.max_duration}s, "
                          f"removing additional step...")

            # Находим оставшиеся шаги (не удаленные)
            remaining_steps = [s for s in steps if not any(
                s['start'] >= rm_start and s['end'] <= rm_end
                for rm_start, rm_end in remove_ranges
            )]

            if not remaining_steps:
                break

            # Удаляем худший из оставшихся
            worst_remaining = min(remaining_steps, key=lambda x: x['avg_score'])
            remove_ranges.append((worst_remaining['start'], worst_remaining['end']))

            # Пересчитываем
            remove_ranges = sorted(remove_ranges, key=lambda x: x[0])
            keep_ranges = self._compute_keep_ranges(window_start, window_duration, remove_ranges)
            final_duration = sum(e - s for s, e in keep_ranges)
            removed_duration = sum(e - s for s, e in remove_ranges)
            iteration += 1

        # Вычисляем средний score ТОЛЬКО для сохраненных сегментов
        kept_segments = []
        for seg in micro_segments:
            seg_start = seg['start']
            seg_end = seg['end']
            # Проверяем попадает ли сегмент в keep_ranges
            is_kept = any(
                not (seg_end <= kr_start or seg_start >= kr_end)
                for kr_start, kr_end in keep_ranges
            )
            if is_kept:
                kept_segments.append(seg)

        avg_score = np.mean([s['score'] for s in kept_segments]) if kept_segments else 0.0

        logger.info(f"Final: {final_duration:.1f}s (removed {removed_duration:.1f}s in {len(remove_ranges)} cuts), "
                   f"avg_score={avg_score:.1f} (kept {len(kept_segments)}/{len(micro_segments)} segments)")

        return {
            'keep_ranges': keep_ranges,
            'remove_ranges': remove_ranges,
            'final_duration': final_duration,
            'removed_duration': removed_duration,
            'avg_score': avg_score
        }

    def _create_steps(
        self,
        micro_segments: List[Dict],
        window_start: float,
        window_duration: float
    ) -> List[Dict]:
        """
        Группирует микро-сегменты в шаги фиксированного размера.
        """
        steps = []
        num_steps = int(np.ceil(window_duration / self.step_size))

        for i in range(num_steps):
            step_start = window_start + i * self.step_size
            step_end = min(step_start + self.step_size, window_start + window_duration)

            # Находим микро-сегменты, попадающие в этот шаг
            step_segments = []
            for seg in micro_segments:
                seg_start = seg['start']
                seg_end = seg['end']

                # Проверяем пересечение
                if not (seg_end <= step_start or seg_start >= step_end):
                    step_segments.append(seg)

            if step_segments:
                avg_score = np.mean([s['score'] for s in step_segments])
            else:
                avg_score = 50.0  # Нейтральный score

            steps.append({
                'index': i,
                'start': step_start,
                'end': step_end,
                'duration': step_end - step_start,
                'avg_score': avg_score,
                'segments': step_segments
            })

        return steps

    def _find_worst_steps(self, steps: List[Dict], count: int) -> List[Dict]:
        """
        Находит N худших шагов по вирусности.
        """
        sorted_steps = sorted(steps, key=lambda x: x['avg_score'])
        return sorted_steps[:count]

    def _group_consecutive_steps(self, steps: List[Dict]) -> List[List[Dict]]:
        """
        Группирует соседние шаги (расстояние = 1).
        """
        if not steps:
            return []

        # Сортируем по индексу
        sorted_steps = sorted(steps, key=lambda x: x['index'])

        groups = []
        current_group = [sorted_steps[0]]

        for i in range(1, len(sorted_steps)):
            prev_step = sorted_steps[i - 1]
            curr_step = sorted_steps[i]

            # Если индексы соседние - добавляем в группу
            if curr_step['index'] - prev_step['index'] == 1:
                current_group.append(curr_step)
            else:
                groups.append(current_group)
                current_group = [curr_step]

        groups.append(current_group)
        return groups

    def _adjust_boundaries_for_speech(
        self,
        step_group: List[Dict],
        speech_segments: List[Dict[str, float]],
        window_start: float,
        window_end: float
    ) -> Tuple[float, float]:
        """
        Корректирует границы удаляемого фрагмента по паузам речи.
        Ищет паузы в пределах ±step_size от границ.
        """
        group_start = step_group[0]['start']
        group_end = step_group[-1]['end']

        # Корректируем начальную границу
        search_start = max(window_start, group_start - self.step_size)
        search_end = min(window_end, group_start + self.step_size)

        adjusted_start = self._find_best_cut_point(
            speech_segments, group_start, search_start, search_end
        )

        # Корректируем конечную границу
        search_start = max(window_start, group_end - self.step_size)
        search_end = min(window_end, group_end + self.step_size)

        adjusted_end = self._find_best_cut_point(
            speech_segments, group_end, search_start, search_end
        )

        logger.debug(f"Adjusted boundaries: [{group_start:.1f},{group_end:.1f}] -> "
                    f"[{adjusted_start:.1f},{adjusted_end:.1f}]")

        return (adjusted_start, adjusted_end)

    def _find_best_cut_point(
        self,
        speech_segments: List[Dict[str, float]],
        default_point: float,
        search_start: float,
        search_end: float
    ) -> float:
        """
        Находит лучшую точку для обрезки - самую длинную паузу речи.
        """
        if not speech_segments:
            return default_point

        # Находим все паузы речи в диапазоне поиска
        pauses = []

        for i in range(len(speech_segments) - 1):
            pause_start = speech_segments[i]['end']
            pause_end = speech_segments[i + 1]['start']
            pause_duration = pause_end - pause_start

            # Проверяем что пауза в диапазоне поиска
            if pause_start >= search_start and pause_end <= search_end:
                if pause_duration >= self.speech_gap_tolerance:
                    pauses.append({
                        'start': pause_start,
                        'end': pause_end,
                        'duration': pause_duration,
                        'midpoint': (pause_start + pause_end) / 2
                    })

        if pauses:
            # Берем самую длинную паузу
            best_pause = max(pauses, key=lambda x: x['duration'])
            logger.debug(f"Found speech pause: {best_pause['duration']:.3f}s at {best_pause['midpoint']:.1f}s")
            return best_pause['midpoint']
        else:
            # Паузы не нашли - используем дефолтную точку
            return default_point

    def _compute_keep_ranges(
        self,
        window_start: float,
        window_duration: float,
        remove_ranges: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Вычисляет диапазоны для сохранения (инверсия remove_ranges).
        """
        if not remove_ranges:
            return [(window_start, window_start + window_duration)]

        sorted_removals = sorted(remove_ranges, key=lambda x: x[0])
        keep_ranges = []
        current_pos = window_start

        for rm_start, rm_end in sorted_removals:
            if current_pos < rm_start:
                keep_ranges.append((current_pos, rm_start))
            current_pos = rm_end

        window_end = window_start + window_duration
        if current_pos < window_end:
            keep_ranges.append((current_pos, window_end))

        return keep_ranges

    def generate_ffmpeg_filter(self, keep_ranges: List[Tuple[float, float]]) -> str:
        """Генерирует FFmpeg select выражение."""
        if not keep_ranges:
            return "0"

        conditions = [f"between(t,{start},{end})" for start, end in keep_ranges]
        return "+".join(conditions)

    def generate_ffmpeg_audio_filter(self, keep_ranges: List[Tuple[float, float]]) -> str:
        """Генерирует FFmpeg aselect выражение."""
        return self.generate_ffmpeg_filter(keep_ranges)
