"""
SORT (Simple Online and Realtime Tracking) — przepisana implementacja.

Różnice względem oryginału (abewley/sort, 2016):
  * brak filterpy — własny filtr Kalmana, z BATCHOWANĄ predykcją i korekcją:
    N tracków to kilka operacji macierzowych, a nie N wywołań w Pythonie
  * parametryzacja stanu (cx, cy, w, h) zamiast (cx, cy, s, r) — patrz BoT-SORT;
    znika problem sqrt(ujemne pole) i źle dopasowanych boxów
  * szum procesowy i pomiarowy skalowany rozmiarem obiektu zamiast stałego Q
  * asocjacja Hungarianem (scipy) zamiast greedy
  * IoU liczone macierzowo, bez pętli po Pythonie
  * spójna obsługa class_id (bramkowanie + głosowanie ważone confidence)
  * zatrzaskiwane potwierdzenie tracku i jawna flaga predykcji w wyjściu

Format wejścia:  (N, 6) -> [x1, y1, x2, y2, conf, class_id]
Format wyjścia:  (M, 8) -> [x1, y1, x2, y2, conf, track_id, class_id, is_predicted]

Zależności: numpy, scipy.
"""

from __future__ import annotations

import numpy as np
from config import Config
from custom_logger import CustomLogger
from scipy.optimize import linear_sum_assignment

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=Config.LOGS_PATH
).create_logger()

# ----------------------------------------------------------------------------
# Konwersje boxów (wszystkie wektorowe)
# ----------------------------------------------------------------------------


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) [x1, y1, x2, y2] -> (N, 4) [cx, cy, w, h]."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    out[:, 0] = boxes[:, 0] + out[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] + out[:, 3] / 2.0
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(N, 4) [cx, cy, w, h] -> (N, 4) [x1, y1, x2, y2]."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    # Przy długiej ekstrapolacji prędkość vw/vh potrafi wypchnąć w/h poniżej zera
    # i box wychodzi odwrócony. Oryginał robił tu sqrt(s*r) i dostawał NaN.
    w = np.maximum(boxes[:, 2], 1e-6)
    h = np.maximum(boxes[:, 3], 1e-6)
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - w / 2.0
    out[:, 1] = boxes[:, 1] - h / 2.0
    out[:, 2] = boxes[:, 0] + w / 2.0
    out[:, 3] = boxes[:, 1] + h / 2.0
    return out


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Macierz IoU między dwoma zbiorami boxów [x1, y1, x2, y2].

    Args:
        boxes_a: (N, 4)
        boxes_b: (M, 4)

    Returns:
        (N, M) macierz IoU
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0.0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0.0, None)
    inter = inter_w * inter_h

    area_a = np.clip(a[..., 2] - a[..., 0], 0.0, None) * np.clip(a[..., 3] - a[..., 1], 0.0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0.0, None) * np.clip(b[..., 3] - b[..., 1], 0.0, None)

    # epsilon w mianowniku — dwa zdegenerowane boxy nie mogą dać NaN,
    # bo NaN zatruwa całą macierz kosztów i psuje przypisanie
    return inter / (area_a + area_b - inter + 1e-9)


# ----------------------------------------------------------------------------
# Filtr Kalmana
# ----------------------------------------------------------------------------


class KalmanFilterXYWH:
    """
    Filtr Kalmana o stałej prędkości dla boxów (cx, cy, w, h).

    Stan 8-wymiarowy: [cx, cy, w, h, vx, vy, vw, vh].
    Pomiar 4-wymiarowy: [cx, cy, w, h].

    Dwie rzeczy, których SORT nie robił:

    1. Szum nie jest stały, tylko proporcjonalny do rozmiaru obiektu. Box
       o wysokości 20 px i box o wysokości 400 px nie mają tej samej niepewności
       położenia — stałe Q udaje, że mają.
    2. Wszystko liczy się dla całej paczki tracków naraz. To jedyny powód, dla
       którego warto pisać ten filtr samemu zamiast brać filterpy: filterpy
       wymusza predict()/update() per obiekt.
    """

    _ndim = 4

    def __init__(self, std_position: float = 1.0 / 20, std_velocity: float = 1.0 / 160):
        ndim, dt = self._ndim, 1.0
        self._motion_mat = np.eye(2 * ndim)
        self._motion_mat[:ndim, ndim:] = dt * np.eye(ndim)
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._eye_state = np.eye(2 * ndim)
        self._eye_meas = np.eye(ndim)

        self.std_position = std_position
        self.std_velocity = std_velocity

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Zainicjuj track z pojedynczego pomiaru [cx, cy, w, h]."""
        mean = np.concatenate([measurement, np.zeros(self._ndim)])
        w, h = measurement[2], measurement[3]
        std = np.array([
            2 * self.std_position * w,
            2 * self.std_position * h,
            2 * self.std_position * w,
            2 * self.std_position * h,
            10 * self.std_velocity * w,
            10 * self.std_velocity * h,
            10 * self.std_velocity * w,
            10 * self.std_velocity * h,
        ])
        return mean, np.diag(np.square(std))

    def multi_predict(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Krok predykcji dla wszystkich tracków naraz.

        Args:
            means: (N, 8)
            covariances: (N, 8, 8)
        """
        if len(means) == 0:
            return means, covariances

        w, h = means[:, 2], means[:, 3]
        std = np.stack([
            self.std_position * w, self.std_position * h,
            self.std_position * w, self.std_position * h,
            self.std_velocity * w, self.std_velocity * h,
            self.std_velocity * w, self.std_velocity * h,
        ], axis=1)
        # (N, 8) -> (N, 8, 8) macierze diagonalne, bez pętli
        motion_cov = np.square(std)[:, :, None] * self._eye_state[None, :, :]

        means = means @ self._motion_mat.T
        covariances = self._motion_mat @ covariances @ self._motion_mat.T + motion_cov
        return means, covariances

    def multi_update(
        self, means: np.ndarray, covariances: np.ndarray, measurements: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Krok korekcji dla wszystkich dopasowanych tracków naraz.

        Wzmocnienie liczone przez np.linalg.solve zamiast jawnej odwrotności —
        stabilniej numerycznie i solve broadcastuje po wymiarze paczki.

        Args:
            means: (N, 8)
            covariances: (N, 8, 8)
            measurements: (N, 4)
        """
        if len(means) == 0:
            return means, covariances

        H = self._update_mat
        w, h = means[:, 2], means[:, 3]
        std = np.stack([
            self.std_position * w, self.std_position * h,
            self.std_position * w, self.std_position * h,
        ], axis=1)
        innovation_cov = np.square(std)[:, :, None] * self._eye_meas[None, :, :]

        projected_mean = means @ H.T                                  # (N, 4)
        projected_cov = H @ covariances @ H.T + innovation_cov        # (N, 4, 4)

        # K = P Hᵀ S⁻¹  <=>  S Kᵀ = (P Hᵀ)ᵀ   (S symetryczna)
        pht = covariances @ H.T                                       # (N, 8, 4)
        gain = np.linalg.solve(projected_cov, pht.transpose(0, 2, 1)).transpose(0, 2, 1)

        innovation = measurements - projected_mean                    # (N, 4)
        new_means = means + np.einsum("nij,nj->ni", gain, innovation)
        new_covs = covariances - gain @ projected_cov @ gain.transpose(0, 2, 1)
        return new_means, new_covs


# ----------------------------------------------------------------------------
# Pojedynczy track
# ----------------------------------------------------------------------------


class Track:
    """
    Pojedyncza hipoteza o obiekcie.

    Trzyma tylko stan i księgowość — arytmetykę filtra robi Sort zbiorczo.
    Wszystko wewnątrz jest w formacie (cx, cy, w, h); konwersje są na granicy.
    """

    _count: int = 0

    def __init__(self, measurement: np.ndarray, conf: float, class_id: int, kf: KalmanFilterXYWH):
        self.mean, self.covariance = kf.initiate(np.asarray(measurement, dtype=np.float64))

        Track._count += 1
        self.id: int = Track._count

        self.conf: float = float(conf)
        # Głosowanie ważone confidence zamiast klasy zamrożonej w pierwszej klatce.
        self._class_votes: dict[int, float] = {int(class_id): float(conf)}

        self.time_since_update: int = 0
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 0
        # Zatrzask. W oryginalnym SORT hit_streak jest zerowany po każdej nieudanej
        # klatce, więc track po każdej luce znika z wyjścia na min_hits klatek —
        # z punktu widzenia konsumenta wygląda to jak utrata i nadanie nowego ID.
        self.confirmed: bool = False

    @property
    def class_id(self) -> int:
        return max(self._class_votes, key=self._class_votes.__getitem__)

    def mark_missed(self) -> None:
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

    def mark_hit(self, conf: float, class_id: int) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.conf = float(conf)
        self._class_votes[int(class_id)] = self._class_votes.get(int(class_id), 0.0) + float(conf)

    @staticmethod
    def reset_id_counter() -> None:
        """Wołaj między sekwencjami wideo, inaczej ID rosną w nieskończoność."""
        Track._count = 0


# ----------------------------------------------------------------------------
# Tracker
# ----------------------------------------------------------------------------


class Sort:
    """
    Args:
        max_age: ile klatek bez detekcji track przeżywa, zanim zostanie usunięty
        min_hits: ile trafień potrzeba, zanim track trafi na wyjście
        iou_threshold: minimalne IoU uznawane za dopasowanie
        per_class: jeśli True, detekcja dopasuje się tylko do tracku tej samej klasy
        emit_predicted: jeśli True, tracki bez detekcji w bieżącej klatce też trafiają
            na wyjście, oznaczone is_predicted=1 — przydatne przy okluzjach, ale
            konsument MUSI tę flagę czytać, bo to ekstrapolacja, nie pomiar
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        per_class: bool = True,
        emit_predicted: bool = False,
        kf: KalmanFilterXYWH | None = None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.per_class = per_class
        self.emit_predicted = emit_predicted

        logger.info("Init EmotionTimeline with: \n"\
                            f"{self.max_age=}\n" \
                            f"{self.min_hits=}\n" \
                            f"{self.iou_threshold=}\n" \
                            f"{self.per_class=}\n" \
                            f"{self.emit_predicted=}\n" \
                    )

        self.kf = kf if kf is not None else KalmanFilterXYWH()
        self.tracks: list[Track] = []
        self.frame_count: int = 0

    def reset(self) -> None:
        """Wyczyść stan i licznik ID — przed nową sekwencją."""
        self.tracks = []
        self.frame_count = 0
        Track.reset_id_counter()

    def _track_boxes(self) -> np.ndarray:
        """Boxy wszystkich tracków jako (N, 4) xyxy — jedna konwersja na klatkę."""
        if not self.tracks:
            return np.empty((0, 4))
        return xywh_to_xyxy(np.stack([t.mean[:4] for t in self.tracks]))

    def update(self, dets: np.ndarray | list | None = None) -> np.ndarray:
        """
        Wołaj RAZ NA KLATKĘ, także wtedy, gdy nie ma żadnych detekcji —
        inaczej tracki nie starzeją się i nie wygasają.

        Args:
            dets: (N, 6) [x1, y1, x2, y2, conf, class_id], albo None / []

        Returns:
            (M, 8) [x1, y1, x2, y2, conf, track_id, class_id, is_predicted]
        """
        self.frame_count += 1

        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float64)
        else:
            dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)

        # --- predykcja (cała paczka jednym mnożeniem) --------------------------
        if self.tracks:
            means = np.stack([t.mean for t in self.tracks])
            covs = np.stack([t.covariance for t in self.tracks])
            means, covs = self.kf.multi_predict(means, covs)
            for t, m, c in zip(self.tracks, means, covs):
                t.mean, t.covariance = m, c
                t.mark_missed()

        # --- asocjacja ---------------------------------------------------------
        matched, unmatched_dets = self._associate(dets)

        # --- korekcja (też jedną paczką) --------------------------------------
        if matched:
            det_idx = np.array([d for d, _ in matched])
            trk_idx = [t for _, t in matched]
            measurements = xyxy_to_xywh(dets[det_idx, :4])

            means = np.stack([self.tracks[i].mean for i in trk_idx])
            covs = np.stack([self.tracks[i].covariance for i in trk_idx])
            means, covs = self.kf.multi_update(means, covs, measurements)

            for i, d, m, c in zip(trk_idx, det_idx, means, covs):
                trk = self.tracks[i]
                trk.mean, trk.covariance = m, c
                trk.mark_hit(dets[d, 4], int(dets[d, 5]))

        # --- nowe tracki -------------------------------------------------------
        if len(unmatched_dets):
            new_meas = xyxy_to_xywh(dets[unmatched_dets, :4])
            for meas, d in zip(new_meas, unmatched_dets):
                self.tracks.append(Track(meas, dets[d, 4], int(dets[d, 5]), self.kf))

        # --- wyjście i sprzątanie ---------------------------------------------
        boxes = self._track_boxes()
        results: list[list[float]] = []
        surviving: list[Track] = []

        for t, box in zip(self.tracks, boxes):
            if not t.confirmed and (
                t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                t.confirmed = True

            measured = t.time_since_update == 0
            if t.confirmed and (measured or self.emit_predicted):
                results.append([
                    box[0], box[1], box[2], box[3],
                    t.conf,
                    float(t.id),
                    float(t.class_id),
                    0.0 if measured else 1.0,
                ])

            if t.time_since_update <= self.max_age:
                surviving.append(t)

        self.tracks = surviving
        return np.array(results, dtype=np.float64) if results else np.empty((0, 8))

    def _associate(self, dets: np.ndarray) -> tuple[list[tuple[int, int]], list[int]]:
        """Przypisanie węgierskie na macierzy IoU, z bramkowaniem progiem i klasą."""
        if len(self.tracks) == 0 or len(dets) == 0:
            return [], list(range(len(dets)))

        iou_matrix = iou_batch(dets[:, :4], self._track_boxes())

        if self.per_class:
            det_cls = dets[:, 5][:, None]
            trk_cls = np.array([t.class_id for t in self.tracks], dtype=np.float64)[None, :]
            iou_matrix = np.where(det_cls == trk_cls, iou_matrix, 0.0)

        # Hungarian minimalizuje koszt, więc maksymalizujemy IoU przez znak minus
        row_idx, col_idx = linear_sum_assignment(-iou_matrix)

        # Hungarian przypisze wszystko, co się da, także pary o IoU = 0 —
        # próg odsiewamy dopiero po fakcie
        keep = iou_matrix[row_idx, col_idx] >= self.iou_threshold
        matched = [(int(r), int(c)) for r, c in zip(row_idx[keep], col_idx[keep])]

        matched_dets = {d for d, _ in matched}
        unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]
        return matched, unmatched_dets


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)

    frames = [
        [[10, 10, 50, 50, 0.90, 0], [100, 100, 150, 150, 0.80, 1]],
        [[12, 12, 52, 52, 0.95, 0], [99, 99, 149, 149, 0.85, 1]],
        [[14, 14, 54, 54, 0.92, 0]],                                # obiekt 2 znika
        [[16, 16, 56, 56, 0.93, 0], [97, 97, 147, 147, 0.70, 1]],   # i wraca
    ]

    for i, dets in enumerate(frames, start=1):
        out = tracker.update(dets)
        print(f"--- klatka {i} ---")
        print("   x1     y1     x2     y2   conf    id   cls  pred")
        print(out if len(out) else "(brak)")
        print()
