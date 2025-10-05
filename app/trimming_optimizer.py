import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrimmingOptimizer:
    """Оптимизатор обрезки видео для получения самых вирусных 60 секунд"""

    def __init__(
        self,
        target_duration: float = 60.0,
        min_remove_duration: float = 5.0,
        max_remove_duration: float = 15.0,
        speech_boundary_tolerance: float = 2.0  # Увеличено с 0.5 до 2.0 для меньшей строгости
    ):
        """
        Args:
            target_duration: Целевая длительность в секундах (60)
            min_remove_duration: Минимальная длительность удаляемого фрагмента
            max_remove_duration: Максимальная длительность удаляемого фрагмента
            speech_boundary_tolerance: Допуск для проверки речи на границах (секунды)
        """
        self.target_duration = target_duration
        self.min_remove_duration = min_remove_duration
        self.max_remove_duration = max_remove_duration
        self.speech_boundary_tolerance = speech_boundary_tolerance
        logger.info(f"TrimmingOptimizer initialized: target={target_duration}s, "
                   f"remove_range=[{min_remove_duration},{max_remove_duration}]s, "
                   f"speech_tolerance={speech_boundary_tolerance}s")

    def optimize_segment(
        self,
        micro_segments: List[Dict],
        speech_segments: List[Dict[str, float]],
        window_start: float,
        window_duration: float,
        is_last_segment: bool = False
    ) -> Dict:
        """
        Оптимизирует 2-минутное окно до 60 секунд, вырезая наименее вирусные фрагменты.

        Args:
            micro_segments: Список детальных сегментов (каждый ~5 сек) с scores
                           [{'start': 0, 'end': 5, 'score': 45}, ...]
            speech_segments: Список сегментов с речью [{'start': 1.5, 'end': 3.2}, ...]
            window_start: Начало окна в секундах
            window_duration: Длительность окна (обычно 120 сек)
            is_last_segment: Является ли это последним сегментом видео

        Returns:
            Dict: {
                'keep_ranges': [(0, 20), (30, 70), ...],  # Диапазоны для сохранения
                'remove_ranges': [(20, 30), ...],         # Диапазоны для удаления
                'final_duration': 60.0,
                'removed_duration': 60.0,
                'avg_score': 75.5
            }
        """
        # Вычисляем сколько нужно удалить
        to_remove = window_duration - self.target_duration

        if to_remove <= 0:
            # Окно уже меньше или равно 60 сек - ничего не удаляем
            return {
                'keep_ranges': [(window_start, window_start + window_duration)],
                'remove_ranges': [],
                'final_duration': window_duration,
                'removed_duration': 0,
                'avg_score': np.mean([s['score'] for s in micro_segments])
            }

        # Сортируем микросегменты по score (от худшего к лучшему)
        sorted_segments = sorted(micro_segments, key=lambda x: x['score'])

        # Ищем кандидатов на удаление
        removal_candidates = self._find_removal_candidates(
            sorted_segments,
            speech_segments,
            window_start,
            window_duration,
            is_last_segment
        )

        # Выбираем оптимальную комбинацию для удаления
        selected_removals = self._select_optimal_removals(
            removal_candidates,
            to_remove
        )

        # Формируем диапазоны для сохранения
        keep_ranges = self._compute_keep_ranges(
            window_start,
            window_duration,
            selected_removals
        )

        # Вычисляем итоговую длительность и средний score
        final_duration = sum(end - start for start, end in keep_ranges)
        removed_duration = sum(end - start for start, end in selected_removals)

        # Средний score оставшихся сегментов
        kept_scores = []
        for segment in micro_segments:
            seg_start = segment['start']
            seg_end = segment['end']
            # Проверяем не удалён ли сегмент
            is_kept = any(
                not (seg_end <= rm_start or seg_start >= rm_end)
                for rm_start, rm_end in selected_removals
            )
            if is_kept or not selected_removals:
                kept_scores.append(segment['score'])

        avg_score = np.mean(kept_scores) if kept_scores else np.mean([s['score'] for s in micro_segments])

        return {
            'keep_ranges': keep_ranges,
            'remove_ranges': selected_removals,
            'final_duration': final_duration,
            'removed_duration': removed_duration,
            'avg_score': float(avg_score)
        }

    def _find_removal_candidates(
        self,
        sorted_segments: List[Dict],
        speech_segments: List[Dict[str, float]],
        window_start: float,
        window_duration: float,
        is_last_segment: bool
    ) -> List[Dict]:
        """
        Находит кандидатов на удаление - последовательные низкорейтинговые сегменты
        без речи на границах.
        """
        candidates = []

        # Группируем последовательные сегменты для удаления
        i = 0
        while i < len(sorted_segments):
            # Начинаем новую группу
            group = [sorted_segments[i]]
            group_start = sorted_segments[i]['start']
            group_end = sorted_segments[i]['end']

            # Пытаемся расширить группу
            j = i + 1
            while j < len(sorted_segments):
                next_seg = sorted_segments[j]

                # Проверяем что сегмент последовательный (примыкает к группе)
                if abs(next_seg['start'] - group_end) < 0.1:  # допуск 0.1 сек
                    potential_end = next_seg['end']
                    potential_duration = potential_end - group_start

                    # Проверяем не превышает ли макс длительность
                    if potential_duration <= self.max_remove_duration:
                        group.append(next_seg)
                        group_end = potential_end
                        j += 1
                    else:
                        break
                else:
                    break

            # Проверяем группу на пригодность к удалению
            duration = group_end - group_start

            if duration >= self.min_remove_duration:
                # Проверка 1: Нет речи на границах
                has_speech_at_start = self._has_speech_near(
                    speech_segments, group_start, self.speech_boundary_tolerance
                )
                has_speech_at_end = self._has_speech_near(
                    speech_segments, group_end, self.speech_boundary_tolerance
                )

                # Исключение: можно обрезать конец видео (последнего сегмента)
                window_end = window_start + window_duration
                is_at_video_end = is_last_segment and abs(group_end - window_end) < 1.0

                # Проверка 2: Не начало и не конец окна (если там речь)
                # Можно удалять:
                # - Если нет речи на обеих границах
                # - Если есть речь на конце, но это конец видео
                can_remove = (
                    not has_speech_at_start and
                    (not has_speech_at_end or is_at_video_end)
                )

                # Логирование отклоненных кандидатов
                if not can_remove:
                    logger.debug(f"Rejected candidate [{group_start:.1f}-{group_end:.1f}]: "
                               f"speech_at_start={has_speech_at_start}, "
                               f"speech_at_end={has_speech_at_end}, "
                               f"is_video_end={is_at_video_end}")

                if can_remove:
                    avg_score = np.mean([s['score'] for s in group])
                    candidates.append({
                        'start': group_start,
                        'end': group_end,
                        'duration': duration,
                        'avg_score': avg_score,
                        'segments_count': len(group),
                        'has_speech': self._has_speech_in_range(
                            speech_segments, group_start, group_end
                        )
                    })
                    logger.debug(f"Accepted candidate [{group_start:.1f}-{group_end:.1f}], "
                               f"duration={duration:.1f}s, avg_score={avg_score:.1f}")

            i = j if j > i else i + 1

        # Если не нашли кандидатов - разрешаем удаление с речью на границах
        # но только для очень низких scores (< 40)
        if len(candidates) == 0:
            logger.info("No speech-safe candidates found, searching for low-score segments...")
            i = 0
            while i < len(sorted_segments):
                group = [sorted_segments[i]]
                group_start = sorted_segments[i]['start']
                group_end = sorted_segments[i]['end']

                j = i + 1
                while j < len(sorted_segments):
                    next_seg = sorted_segments[j]
                    if abs(next_seg['start'] - group_end) < 0.1:
                        potential_end = next_seg['end']
                        potential_duration = potential_end - group_start
                        if potential_duration <= self.max_remove_duration:
                            group.append(next_seg)
                            group_end = potential_end
                            j += 1
                        else:
                            break
                    else:
                        break

                duration = group_end - group_start
                if duration >= self.min_remove_duration:
                    avg_score = np.mean([s['score'] for s in group])
                    # Разрешаем только очень низкие scores
                    if avg_score < 40:
                        candidates.append({
                            'start': group_start,
                            'end': group_end,
                            'duration': duration,
                            'avg_score': avg_score,
                            'segments_count': len(group),
                            'has_speech': True  # Предполагаем что речь есть
                        })
                        logger.info(f"Added low-score candidate [{group_start:.1f}-{group_end:.1f}], score={avg_score:.1f}")

                i = j if j > i else i + 1

        # Сортируем кандидатов по приоритету удаления (худший score, нет речи внутри)
        candidates.sort(key=lambda x: (x['avg_score'], x['has_speech']))

        logger.info(f"Found {len(candidates)} removal candidates (total)")
        return candidates

    def _select_optimal_removals(
        self,
        candidates: List[Dict],
        to_remove: float
    ) -> List[Tuple[float, float]]:
        """
        Выбирает оптимальную комбинацию фрагментов для удаления.
        Жадный алгоритм: берем худшие сегменты пока не достигнем нужной длительности.
        """
        selected = []
        total_removed = 0.0

        for candidate in candidates:
            # Проверяем достигли ли цели (с допуском 10%)
            if total_removed >= to_remove * 0.9:
                break

            # Берем кандидата
            duration = candidate['duration']
            remaining = to_remove - total_removed

            # Берем кандидата если:
            # 1. Он помещается полностью (duration <= remaining + 5 сек допуск)
            # 2. ИЛИ мы еще далеко от цели (< 80% удалено)
            if duration <= remaining + 5.0 or total_removed < to_remove * 0.8:
                selected.append((candidate['start'], candidate['end']))
                total_removed += duration

        logger.info(f"Selected {len(selected)} segments for removal (total: {total_removed:.1f}s)")
        return selected

    def _compute_keep_ranges(
        self,
        window_start: float,
        window_duration: float,
        remove_ranges: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Вычисляет диапазоны которые нужно сохранить (инверсия remove_ranges).
        """
        if not remove_ranges:
            return [(window_start, window_start + window_duration)]

        # Сортируем по началу
        sorted_removals = sorted(remove_ranges, key=lambda x: x[0])

        keep_ranges = []
        current_pos = window_start

        for rm_start, rm_end in sorted_removals:
            # Добавляем промежуток до удаляемого фрагмента
            if current_pos < rm_start:
                keep_ranges.append((current_pos, rm_start))
            current_pos = rm_end

        # Добавляем остаток после последнего удаления
        window_end = window_start + window_duration
        if current_pos < window_end:
            keep_ranges.append((current_pos, window_end))

        return keep_ranges

    def _has_speech_near(
        self,
        speech_segments: List[Dict[str, float]],
        timestamp: float,
        tolerance: float
    ) -> bool:
        """Проверяет есть ли речь около временной метки."""
        for segment in speech_segments:
            if segment['start'] - tolerance <= timestamp <= segment['end'] + tolerance:
                return True
        return False

    def _has_speech_in_range(
        self,
        speech_segments: List[Dict[str, float]],
        start: float,
        end: float
    ) -> bool:
        """Проверяет есть ли речь в диапазоне."""
        for segment in speech_segments:
            # Любое пересечение
            if not (segment['end'] < start or segment['start'] > end):
                return True
        return False

    def generate_ffmpeg_filter(
        self,
        keep_ranges: List[Tuple[float, float]]
    ) -> str:
        """
        Генерирует FFmpeg select выражение для сохранения указанных диапазонов.
        Возвращает только выражение between без select= и setpts=.

        Args:
            keep_ranges: Список диапазонов для сохранения [(0, 20), (30, 70), ...]

        Returns:
            str: between выражение для select фильтра
        """
        if not keep_ranges:
            return "0"  # Пустой результат (ничего не выбрано)

        # Генерируем условия between для каждого диапазона
        conditions = []
        for start, end in keep_ranges:
            conditions.append(f"between(t,{start},{end})")

        return "+".join(conditions)

    def generate_ffmpeg_audio_filter(
        self,
        keep_ranges: List[Tuple[float, float]]
    ) -> str:
        """
        Генерирует FFmpeg aselect выражение для аудио.
        Возвращает только выражение between без aselect= и asetpts=.
        """
        if not keep_ranges:
            return "0"

        conditions = []
        for start, end in keep_ranges:
            conditions.append(f"between(t,{start},{end})")

        return "+".join(conditions)
