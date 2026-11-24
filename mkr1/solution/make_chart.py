"""Будує evidence/charts/before_after.png з виміряних даних (потрібен matplotlib; для запуску інструмента він НЕ потрібен)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import finalize_report as fr  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BEFORE, AFTER = "#2a78d6", "#eb6834"  # слоти 1 і 2 валідованої категоріальної палітри (скрипт validate_palette: ALL CHECKS PASS)


def main() -> None:
    est = fr.load(fr.HERE / "estimates.json") or {}
    tm = fr.load(fr.EV / "timing.json") or {}
    ai = fr.load(fr.EV / "ai_run" / "bench_ai.json")
    hist = fr.history()
    static_s = sum(t["mean_s"] for t in tm.values()) / len(tm) if tm else 0.0
    ai_s = 0.0
    if ai and ai.get("judge"):
        j = ai["judge"]
        ai_s = j["latency_s"] / max(1, j["tokens"]["calls"] + j["tokens"]["cache_hits"]) * 6
    man, read_min = est.get("manual_check_minutes", 20), est.get("reading_output_minutes_after", 2)
    tool_min = (static_s + ai_s) / 60

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1]})
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.tick_params(length=0)

    # --- панель 1: час на перевірку однієї лаби
    a1.bar(0, man, width=0.5, color=BEFORE, hatch="////", edgecolor=SURFACE, linewidth=1.5)
    a1.bar(1, read_min, width=0.5, color=AFTER, hatch="////", edgecolor=SURFACE, linewidth=1.5)
    a1.bar(1, tool_min, width=0.5, bottom=read_min + 0.12, color=AFTER, edgecolor=SURFACE, linewidth=1.5)
    a1.text(0, man + 0.5, f"≈ {man:g} хв".replace(".", ","), ha="center", color=INK, fontsize=11, fontweight="bold")
    a1.text(1, read_min + tool_min + 2.6, f"≈ {read_min + tool_min:.1f} хв".replace(".", ","), ha="center", color=INK, fontsize=11, fontweight="bold")
    a1.text(1, read_min + tool_min + 0.8, f"із них інструмент {static_s + ai_s:.1f} с (виміряно)".replace(".", ","), ha="center", color=INK2, fontsize=9)
    a1.set_xticks([0, 1], ["До: вручну", "Після: labcheck"], fontsize=10)
    a1.set_ylim(0, man * 1.25)
    a1.set_ylabel("хвилин на одну лабу")
    a1.set_title("Час перевірки перед здачею", loc="left", color=INK, fontsize=12, fontweight="bold")
    handles = [Patch(facecolor=INK2, hatch="////", edgecolor=SURFACE, label="оцінка без таймера (з голови)")]
    if tool_min > 0.1:
        handles.append(Patch(facecolor=INK2, edgecolor=SURFACE, label="виміряно (час роботи інструмента)"))
    a1.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)

    # --- панель 2: незакриті плейсхолдери в реальних ноутбуках
    labs = [("Лаба 5", "lab5"), ("Лаба 6", "lab6")]
    xs = range(len(labs))
    bw = 0.32
    for i, (name, key) in enumerate(labs):
        b, a = fr.placeholders_in_log(f"{key}_before"), fr.placeholders_in_log(f"{key}_after")
        a2.bar(i - bw / 2 - 0.01, b, width=bw, color=BEFORE, edgecolor=SURFACE, linewidth=1.5)
        a2.bar(i + bw / 2 + 0.01, a, width=bw, color=AFTER, edgecolor=SURFACE, linewidth=1.5)
        a2.text(i - bw / 2 - 0.01, b + 0.8, str(b), ha="center", color=INK, fontsize=11, fontweight="bold")
        a2.text(i + bw / 2 + 0.01, a + 0.8, str(a), ha="center", color=INK, fontsize=11, fontweight="bold")
    a2.set_xticks(list(xs), [n for n, _ in labs], fontsize=10)
    a2.set_ylim(0, max(5, fr.placeholders_in_log("lab6_before") * 1.25))
    a2.set_ylabel("клітинок із плейсхолдером")
    a2.set_title("Незакриті плейсхолдери в ваших ноутбуках", loc="left", color=INK, fontsize=12, fontweight="bold")
    a2.legend(handles=[Patch(facecolor=BEFORE, label="до доопрацювання (ноутбуки зі Colab)"), Patch(facecolor=AFTER, label="після (фінальна версія)")],
              loc="upper left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.text(0.01, 0.01, "Панель 2 — реальні прогони labcheck, див. evidence/real_runs/*.txt. Заштриховане — оцінка, не вимірювання.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = fr.EV / "charts"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "before_after.png", dpi=160, facecolor=SURFACE)
    print("збережено", out / "before_after.png")


if __name__ == "__main__":
    main()
