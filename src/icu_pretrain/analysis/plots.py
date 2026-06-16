from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

from icu_pretrain.analysis.tables import DEFAULT_FIGURES_DIR, DEFAULT_RUN_ROOT, DEFAULT_SUMMARY_DIR, load_report_payload

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


PALETTE = [
    (78 / 255, 121 / 255, 167 / 255),
    (242 / 255, 142 / 255, 43 / 255),
    (225 / 255, 87 / 255, 89 / 255),
    (118 / 255, 183 / 255, 178 / 255),
]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(data)
    temp_path.replace(path)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _encode_png(width: int, height: int, pixels: bytearray) -> bytes:
    rows = bytearray()
    stride = width * 3
    for y in range(height):
        rows.append(0)
        start = y * stride
        rows.extend(pixels[start : start + stride])
    compressed = zlib.compress(bytes(rows), level=9)
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return header + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")


def _canvas(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)) -> bytearray:
    pixels = bytearray(width * height * 3)
    for idx in range(0, len(pixels), 3):
        pixels[idx] = color[0]
        pixels[idx + 1] = color[1]
        pixels[idx + 2] = color[2]
    return pixels


def _set_pixel(pixels: bytearray, width: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0:
        return
    height = len(pixels) // (width * 3)
    if x >= width or y >= height:
        return
    idx = (y * width + x) * 3
    pixels[idx] = color[0]
    pixels[idx + 1] = color[1]
    pixels[idx + 2] = color[2]


def _fill_rect(pixels: bytearray, width: int, left: int, top: int, right: int, bottom: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, top), max(0, bottom)):
        for x in range(max(0, left), max(0, right)):
            _set_pixel(pixels, width, x, y, color)


def _line(pixels: bytearray, width: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        _set_pixel(pixels, width, x, y, color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _render_grouped_bars(
    path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float], tuple[float, float, float]]],
    ylabel: str,
) -> None:
    if plt is not None:
        figure, axis = plt.subplots(figsize=(11, 6))
        if categories and series:
            x_positions = range(len(categories))
            bar_width = 0.8 / max(1, len(series))
            offsets = [index - 0.4 + bar_width / 2 for index in x_positions]
            for series_index, (series_name, values, color) in enumerate(series):
                positions = [offset + series_index * bar_width for offset in offsets]
                axis.bar(positions, values, width=bar_width, label=series_name, color=color)
            axis.set_xticks(list(x_positions))
            axis.set_xticklabels(categories, rotation=20, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.legend(frameon=False)
            axis.grid(axis="y", alpha=0.2)
        else:
            axis.text(0.5, 0.5, "No data available", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return

    width = 1100
    height = 650
    pixels = _canvas(width, height)
    if not categories or not series:
        _fill_rect(pixels, width, 120, 120, width - 120, height - 120, (240, 240, 240))
        _atomic_write_bytes(path, _encode_png(width, height, pixels))
        return

    left = 90
    right = width - 40
    top = 60
    bottom = height - 80
    chart_width = right - left
    chart_height = bottom - top
    max_value = max((max(values) if values else 0.0) for _, values, _ in series)
    if max_value <= 0:
        max_value = 1.0

    _line(pixels, width, left, top, left, bottom, (0, 0, 0))
    _line(pixels, width, left, bottom, right, bottom, (0, 0, 0))
    group_width = chart_width / max(1, len(categories))
    bar_width = group_width / (len(series) + 1)

    for category_index, _ in enumerate(categories):
        group_left = left + category_index * group_width
        for series_index, (_, values, color) in enumerate(series):
            value = values[category_index] if category_index < len(values) else 0.0
            bar_height = int((value / max_value) * (chart_height - 10))
            bar_left = int(group_left + (series_index + 0.5) * bar_width)
            bar_right = int(bar_left + bar_width * 0.8)
            bar_top = bottom - bar_height
            rgb = tuple(int(channel * 255) for channel in color)
            _fill_rect(pixels, width, bar_left, bar_top, bar_right, bottom, rgb)

    _atomic_write_bytes(path, _encode_png(width, height, pixels))


def _extract_metric_series(rows: list[dict[str, Any]], key: str) -> list[float]:
    series = []
    for row in rows:
        value = row.get(key, "")
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if number != number:
            number = 0.0
        series.append(number)
    return series


def _round_value(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:
        return 0.0
    return number


def _save_model_comparison(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["selected_model"]
    categories = [str(row.get("experiment_id", "")) for row in rows] or ["no-data"]
    series = [
        ("AUROC", [_round_value(row.get("auroc", 0.0)) for row in rows] or [0.0], PALETTE[0]),
        ("AP", [_round_value(row.get("average_precision", 0.0)) for row in rows] or [0.0], PALETTE[1]),
    ]
    _render_grouped_bars(path, "Model comparison", categories, series, "metric")


def _save_hospital_grouped(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["hospital_grouped_runs"]
    if rows:
        categories = [str(item["run_id"]) for item in rows]
        series = [
            ("AUROC", [_round_value(item["summary"].get("pooled", {}).get("auroc", 0.0)) for item in rows], PALETTE[0]),
            ("AP", [_round_value(item["summary"].get("pooled", {}).get("average_precision", 0.0)) for item in rows], PALETTE[1]),
        ]
    else:
        categories = []
        series = []
    _render_grouped_bars(path, "Hospital-grouped performance", categories or ["no-data"], series or [("AUROC", [0.0], PALETTE[0])], "metric")


def _save_fedavg(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["fedavg_runs"]
    if rows:
        categories = [str(item["run_id"]) for item in rows]
        central = [_round_value(item["summary"].get("central_reference", {}).get("auroc", 0.0)) for item in rows]
        federated = [_round_value(item["summary"].get("federated", {}).get("auroc", 0.0)) for item in rows]
        series = [
            ("Central AUROC", central, PALETTE[0]),
            ("FedAvg AUROC", federated, PALETTE[2]),
        ]
    else:
        categories = []
        series = []
    _render_grouped_bars(path, "FedAvg versus centralised", categories or ["no-data"], series or [("Central AUROC", [0.0], PALETTE[0])], "auroc")


def make_plots(
    summary_dir: Path | None = None,
    run_root: Path | None = None,
    figures_dir: Path | None = None,
) -> list[Path]:
    summary_dir = Path(summary_dir or DEFAULT_SUMMARY_DIR)
    run_root = Path(run_root or DEFAULT_RUN_ROOT)
    figures_dir = Path(figures_dir or DEFAULT_FIGURES_DIR)
    payload = load_report_payload(summary_dir=summary_dir, run_root=run_root)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures_dir / "model_comparison.png",
        figures_dir / "hospital_grouped_performance.png",
        figures_dir / "fedavg_vs_centralised.png",
    ]
    _save_model_comparison(outputs[0], payload)
    _save_hospital_grouped(outputs[1], payload)
    _save_fedavg(outputs[2], payload)
    return outputs
