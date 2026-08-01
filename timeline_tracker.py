"""Agregacja emocji w przedzialy czasowe, odporna na zmienna przepustowosc petli."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from statistics import median


@dataclass
class EmotionInterval:
    emotion: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _OpenInterval:
    emotion: str
    start: float
    end: float


class EmotionTimeline:
    """
    Zamienia strumien (obj_id, emocja, conf, timestamp) na liste przedzialow czasowych.

    Parametry sa w SEKUNDACH, nie w klatkach - dzieki temu dzialaja tak samo
    przy 30 klatkach na sekunde i przy 0.5, bez recznego strojenia.

    smoothing_seconds - dlugosc okna glosowania (stabilizacja predykcji CNN)
    min_votes         - tyle ostatnich probek zostaje w oknie nawet gdy sa starsze
    min_conf          - predykcje ponizej tej pewnosci nie glosuja
    min_duration      - przedzialy krotsze niz tyle sekund sa wyrzucane
    max_gap           - None = auto (gap_factor x zmierzony odstep miedzy probkami)
    gap_factor        - ile odstepow bez twarzy oznacza, ze zniknela z kadru
    """

    def __init__(self,
                 smoothing_seconds: float = 0.3,
                 min_votes: int = 3,
                 min_conf: float = 0.4,
                 min_duration: float = 0.4,
                 max_gap: float | None = None,
                 gap_factor: float = 3.0) -> None:
        self.smoothing_seconds = smoothing_seconds
        self.min_votes = max(1, min_votes)
        self.min_conf = min_conf
        self.min_duration = min_duration
        self.max_gap = max_gap
        self.gap_factor = gap_factor

        self._deltas: deque[float] = deque(maxlen=31)
        self._last_ts: float | None = None

        self._votes: dict[int, deque] = defaultdict(deque)
        self._open: dict[int, _OpenInterval] = {}
        self.timelines: dict[int, list[EmotionInterval]] = defaultdict(list)

    @property
    def sampling_interval(self) -> float | None:
        """Zmierzony odstep miedzy przetworzonymi klatkami (mediana, odporna na zacieci)."""
        return median(self._deltas) if len(self._deltas) >= 3 else None

    @property
    def effective_fps(self) -> float | None:
        si = self.sampling_interval
        return 1 / si if si else None

    @property
    def effective_max_gap(self) -> float:
        if self.max_gap is not None:
            return self.max_gap
        si = self.sampling_interval
        return max(self.gap_factor * si, 2 * self.min_duration) if si else float("inf")

    def _note_timestamp(self, timestamp: float) -> None:
        """Mierzy odstepy miedzy KLATKAMI, nie miedzy twarzami na jednej klatce."""
        if self._last_ts is None:
            self._last_ts = timestamp
        elif timestamp > self._last_ts:
            self._deltas.append(timestamp - self._last_ts)
            self._last_ts = timestamp

    def update(self, obj_id: int, emotion: str, conf: float, timestamp: float) -> str | None:
        """Wywoluj raz na klatke dla kazdej wykrytej twarzy. Zwraca wygladzona emocje."""
        self._note_timestamp(timestamp)

        votes = self._votes[obj_id]
        if conf >= self.min_conf:
            votes.append((timestamp, emotion))

        cutoff = timestamp - self.smoothing_seconds
        while len(votes) > self.min_votes and votes[0][0] < cutoff:
            votes.popleft()
        if not votes:
            return None

        stable = Counter(e for _, e in votes).most_common(1)[0][0]
        current = self._open.get(obj_id)

        if current is None:
            self._open[obj_id] = _OpenInterval(stable, timestamp, timestamp)
        elif timestamp - current.end > self.effective_max_gap:
            self._close(obj_id, current.end, pad=True)   # twarz zniknela z kadru
            self._open[obj_id] = _OpenInterval(stable, timestamp, timestamp)
        elif stable != current.emotion:
            self._close(obj_id, timestamp)               # przedzialy stykaja sie ze soba
            self._open[obj_id] = _OpenInterval(stable, timestamp, timestamp)
        else:
            current.end = timestamp

        return stable

    def finalize(self, timestamp: float) -> None:
        for obj_id in list(self._open):
            self._close(obj_id, max(timestamp, self._open[obj_id].end), pad=True)

    def _close(self, obj_id: int, end: float, pad: bool = False) -> None:
        current = self._open.pop(obj_id, None)
        if current is None:
            return

        # Probka "pokrywa" czas do nastepnej spodziewanej probki - inaczej przy wolnym
        # przetwarzaniu pojedyncza obserwacja mialaby dlugosc 0 i zniknelaby z raportu.
        if pad and end <= current.start:
            end = current.start + (self.sampling_interval or 0)

        interval = EmotionInterval(current.emotion, current.start, end)
        if interval.duration < self.min_duration:
            return

        history = self.timelines[obj_id]
        if history:
            prev = history[-1]
            if prev.emotion == interval.emotion and interval.start - prev.end <= self.effective_max_gap:
                prev.end = interval.end
                return
        history.append(interval)

    def per_emotion(self, obj_id: int) -> dict[str, list[EmotionInterval]]:
        grouped: dict[str, list[EmotionInterval]] = defaultdict(list)
        for interval in self.timelines[obj_id]:
            grouped[interval.emotion].append(interval)
        return grouped

    def to_dict(self) -> dict:
        return {
            obj_id: {
                emotion: [
                    {"start": round(i.start, 2),
                     "end": round(i.end, 2),
                     "duration": round(i.duration, 2)}
                    for i in intervals
                ]
                for emotion, intervals in self.per_emotion(obj_id).items()
            }
            for obj_id in sorted(self.timelines)
        }

    def report(self) -> str:
        lines = []
        if self.effective_fps:
            lines.append(f"[przetworzono {self.effective_fps:.1f} klatek/s, "
                         f"max_gap = {self.effective_max_gap:.2f}s]")
        for obj_id in sorted(self.timelines):
            lines.append(f"face{obj_id}:")
            for emotion, intervals in sorted(self.per_emotion(obj_id).items(),
                                             key=lambda kv: sum(i.duration for i in kv[1]),
                                             reverse=True):
                spans = ", ".join(f"{i.start:.1f}-{i.end:.1f}s" for i in intervals)
                total = sum(i.duration for i in intervals)
                lines.append(f"  {emotion}: {spans}   (razem {total:.1f}s)")
        return "\n".join(lines)