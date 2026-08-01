from math import ceil, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend bez GUI - tylko zapis do pliku

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

Report = dict[int, dict[str, list[dict]]]


def _emotion_colors(report: Report) -> dict[str, tuple]:
    emotions = sorted({e for history in report.values() for e in history})
    cmap = plt.get_cmap("tab10" if len(emotions) <= 10 else "tab20")

    return {emotion: cmap(i % cmap.N) for i, emotion in enumerate(emotions)}


def _grid(n: int, max_cols: int = 3) -> tuple[int, int]:
    """grid that is as close to square as possible: 1->1x1, 2->1x2, 3->1x3, 4->2x2, 6->2x3, 9->3x3.."""
    cols = min(max_cols, ceil(sqrt(n)))
    rows = ceil(n / cols)

    return rows, cols


def plot_emotion_totals(report: Report,
                        out_path: str | Path = "emotion_totals.png",
                        max_cols: int = 3) -> Path | None:
    """Bar plot"""
    if not report:
        return None

    colors = _emotion_colors(report)
    faces = sorted(report)
    rows, cols = _grid(len(faces), max_cols)

    fig, axes = plt.subplots(rows, cols,
                             figsize=(5 * cols, 3.6 * rows),
                             squeeze=False)
    flat = axes.ravel()

    for ax, face_id in zip(flat, faces):
        totals = {emotion: sum(i["duration"] for i in intervals)
                  for emotion, intervals in report[face_id].items()}
        totals = dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

        bars = ax.bar(list(totals), list(totals.values()),
                      color=[colors[e] for e in totals])
        ax.bar_label(bars, fmt="%.1f s", padding=2, fontsize=8)
        ax.set_title(f"face{face_id}")
        ax.set_xlabel("Emotion [-]")
        ax.set_ylabel("Time [s]")
        ax.tick_params(axis="x", rotation=30)
        ax.margins(y=.18)
        ax.grid(axis="y", alpha=.3)
        ax.set_axisbelow(True)

    for ax in flat[len(faces):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_emotion_timeline(report: Report,
                          out_path: str | Path = "emotion_timeline.png",
                          video_duration: float | None = None) -> Path | None:
    if not report:
        return None

    colors = _emotion_colors(report)
    faces = sorted(report)

    fig, ax = plt.subplots(figsize=(12, 1.6 + .75 * len(faces)))

    for row, face_id in enumerate(faces):
        for emotion, intervals in report[face_id].items():
            spans = [(i["start"], i["duration"]) for i in intervals]
            ax.broken_barh(spans, (row - .35, .7),
                           facecolors=colors[emotion],
                           edgecolors="white", linewidth=.5)

    ax.set_yticks(range(len(faces)))
    ax.set_yticklabels([f"face{f}" for f in faces])
    ax.set_ylim(-.7, len(faces) - .3)
    ax.invert_yaxis()
    ax.set_xlim(0, video_duration or max(
        i["end"] for history in report.values()
        for intervals in history.values() for i in intervals
    ))
    ax.set_xlabel("Czas filmu [s]")
    ax.grid(axis="x", alpha=.3)
    ax.set_axisbelow(True)

    ax.legend(handles=[Patch(facecolor=c, label=e) for e, c in colors.items()],
              loc="lower center", bbox_to_anchor=(.5, 1.01),
              ncol=min(len(colors), 6), frameon=False)

    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return out_path


def save_plots(report: Report,
               out_dir: str | Path = "plots",
               video_duration: float | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [p for p in (
        plot_emotion_totals(report, out_dir / "emotion_totals.png"),
        plot_emotion_timeline(report, out_dir / "emotion_timeline.png", video_duration),
    ) if p is not None]