#!/usr/bin/env python3
"""Serial collector + live plotter for SonoCorpus glove data.

Expected serial format (from Arduino):
Index\tMiddle\tRing\tThumb
123\t456\t789\t321
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional

import serial
from serial.tools import list_ports


CHANNEL_NAMES = ("Index", "Middle", "Ring", "Thumb")
EXPECTED_HEADER = "\t".join(CHANNEL_NAMES)
DEFAULT_BAUD = 115200
DEFAULT_WINDOW_SEC = 10.0
DEFAULT_RECONNECT_SEC = 1.0
DEFAULT_EMA_ALPHA = 0.2
DEFAULT_PLOT_FPS = 12.0
DEFAULT_YLIM_UPDATE_SEC = 0.35


@dataclass
class Sample:
    timestamp_s: float
    index: int
    middle: int
    ring: int
    thumb: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and live-plot 4-channel glove data from Arduino serial output."
    )
    parser.add_argument("--port", help="Serial device path (auto-detect if omitted).")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate.")
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_WINDOW_SEC,
        help="Plot window in seconds.",
    )
    parser.add_argument(
        "--csv",
        help=(
            "CSV output path. Defaults to data/glove_capture_YYYYMMDD_HHMMSS.csv "
            "unless --no-csv is set."
        ),
    )
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV recording.")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plotting.")
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional duration (seconds) before auto-stop.",
    )
    parser.add_argument(
        "--reconnect-sec",
        type=float,
        default=DEFAULT_RECONNECT_SEC,
        help="Delay between auto-reconnect attempts after serial disconnects.",
    )
    parser.add_argument(
        "--plot-fps",
        type=float,
        default=DEFAULT_PLOT_FPS,
        help="Max plot refresh rate. Lower values reduce serial lag.",
    )
    return parser.parse_args()


def default_csv_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("data", f"glove_capture_{stamp}.csv")


def _score_port(port: list_ports.ListPortInfo) -> int:
    device = port.device or ""
    text = " ".join(s.lower() for s in (device, port.description or "", port.manufacturer or ""))
    score = 0

    keyword_scores = {
        "arduino": 100,
        "sparkfun": 95,
        "usb serial": 60,
        "ttyacm": 80,
        "tty.usbmodem": 75,
        "tty.usbserial": 70,
    }
    for key, value in keyword_scores.items():
        if key in text:
            score += value

    if device.upper().startswith("COM"):
        score += 65
    if device.startswith("/dev/tty."):
        score += 20
    if device.startswith("/dev/cu."):
        score += 10

    if port.vid is not None and port.pid is not None:
        score += 10

    return score


def autodetect_port() -> Optional[str]:
    ports = list(list_ports.comports())
    if not ports:
        return None

    ranked = sorted(ports, key=_score_port, reverse=True)
    if _score_port(ranked[0]) <= 0:
        return None
    return ranked[0].device


def parse_sample_line(line: str) -> Optional[tuple[int, int, int, int]]:
    trimmed = line.strip()
    if not trimmed:
        return None

    if trimmed.lower() == EXPECTED_HEADER.lower():
        return None

    delimiter = "\t" if "\t" in trimmed else ","
    parts = [p.strip() for p in trimmed.split(delimiter)]
    if len(parts) != 4:
        return None

    try:
        index, middle, ring, thumb = (int(p) for p in parts)
    except ValueError:
        return None

    return index, middle, ring, thumb


class Plotter:
    def __init__(self, window_sec: float, plot_fps: float):
        import matplotlib.pyplot as plt

        self.window_sec = max(0.5, window_sec)
        self.ema_alpha = DEFAULT_EMA_ALPHA
        self.ema_state: list[Optional[float]] = [None, None, None, None]
        self.plt = plt
        self.fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 7))
        self.axes = list(axes)
        self.fig.suptitle(f"Glove Piezo Channels (EMA alpha={self.ema_alpha})")

        self.times: Deque[float] = deque()
        self.data = [deque() for _ in range(4)]
        self.lines = []
        for i, name in enumerate(CHANNEL_NAMES):
            ax = self.axes[i]
            line, = ax.plot([], [])
            self.lines.append(line)
            ax.set_ylabel(f"{name}\nADC")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        self.axes[-1].set_xlabel("Time (s)")
        self.fig.tight_layout(rect=(0, 0, 1, 0.97))

        self._last_draw = 0.0
        self.draw_interval_s = 1.0 / max(1.0, plot_fps)
        self._last_ylim_update = 0.0
        self.ylim_update_interval_s = DEFAULT_YLIM_UPDATE_SEC
        self.plt.ion()
        self.fig.show()

    def add(self, sample: Sample) -> None:
        self.times.append(sample.timestamp_s)
        values = (sample.index, sample.middle, sample.ring, sample.thumb)
        for i, value in enumerate(values):
            prev = self.ema_state[i]
            if prev is None:
                ema_value = float(value)
            else:
                ema_value = self.ema_alpha * float(value) + (1.0 - self.ema_alpha) * prev
            self.ema_state[i] = ema_value
            self.data[i].append(ema_value)

        min_time = sample.timestamp_s - self.window_sec
        while self.times and self.times[0] < min_time:
            self.times.popleft()
            for d in self.data:
                d.popleft()

        now = time.monotonic()
        if now - self._last_draw >= self.draw_interval_s:
            self.redraw()
            self._last_draw = now

    def redraw(self) -> None:
        if not self.times:
            return

        x = list(self.times)
        for i, line in enumerate(self.lines):
            line.set_data(x, list(self.data[i]))

        left = max(0.0, x[-1] - self.window_sec)
        right = max(self.window_sec, x[-1])
        now = time.monotonic()
        should_update_ylim = (
            self._last_ylim_update == 0.0
            or (now - self._last_ylim_update) >= self.ylim_update_interval_s
        )
        for i, ax in enumerate(self.axes):
            ax.set_xlim(left, right)
            if should_update_ylim:
                visible_values = list(self.data[i])
                if not visible_values:
                    continue
                y_min = min(visible_values)
                y_max = max(visible_values)
                if y_min == y_max:
                    pad = max(5.0, abs(float(y_min)) * 0.05 + 1.0)
                else:
                    pad = max(5.0, float(y_max - y_min) * 0.08)

                y_low = float(y_min) - pad
                y_high = float(y_max) + pad
                if y_high <= y_low:
                    y_high = y_low + 1.0
                ax.set_ylim(y_low, y_high)
        if should_update_ylim:
            self._last_ylim_update = now

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


class CsvWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp_s", "index", "middle", "ring", "thumb"])
        self._path = path
        self._rows_since_flush = 0

    @property
    def path(self) -> str:
        return self._path

    def write(self, sample: Sample) -> None:
        self._writer.writerow(
            [
                f"{sample.timestamp_s:.6f}",
                sample.index,
                sample.middle,
                sample.ring,
                sample.thumb,
            ]
        )
        self._rows_since_flush += 1
        if self._rows_since_flush >= 20:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        self._file.flush()
        self._file.close()


def main() -> int:
    args = parse_args()

    if args.window_sec <= 0:
        print("Error: --window-sec must be > 0", file=sys.stderr)
        return 2
    if args.duration is not None and args.duration <= 0:
        print("Error: --duration must be > 0", file=sys.stderr)
        return 2
    if args.reconnect_sec <= 0:
        print("Error: --reconnect-sec must be > 0", file=sys.stderr)
        return 2
    if args.plot_fps <= 0:
        print("Error: --plot-fps must be > 0", file=sys.stderr)
        return 2

    fixed_port = args.port
    auto_port = autodetect_port() if fixed_port is None else None
    if fixed_port is None and not auto_port:
        print(
            "Could not auto-detect a serial port. Use --port <device> (e.g. /dev/tty.usbmodemXXXX or COM3).",
            file=sys.stderr,
        )
        return 2

    csv_writer: Optional[CsvWriter] = None
    if not args.no_csv:
        csv_path = args.csv or default_csv_path()
        csv_writer = CsvWriter(csv_path)

    plotter: Optional[Plotter] = None
    if not args.no_plot:
        try:
            plotter = Plotter(args.window_sec, args.plot_fps)
        except Exception as exc:
            if csv_writer:
                csv_writer.close()
            print(f"Failed to initialize plotting backend: {exc}", file=sys.stderr)
            return 1

    stop = False

    def _handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    if csv_writer:
        print(f"Recording CSV: {csv_writer.path}")

    start_time = time.monotonic()
    sample_count = 0
    last_announced_port: Optional[str] = None

    try:
        while not stop:
            elapsed = time.monotonic() - start_time
            if args.duration is not None and elapsed >= args.duration:
                break

            port = fixed_port or autodetect_port()
            if not port:
                print(
                    f"No serial port found. Retrying in {args.reconnect_sec:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(args.reconnect_sec)
                continue

            if port != last_announced_port:
                print(f"Connecting to {port} @ {args.baud} baud...")
                last_announced_port = port

            try:
                with serial.Serial(
                    port=port,
                    baudrate=args.baud,
                    timeout=0.25,
                    exclusive=True,
                    rtscts=False,
                    dsrdtr=False,
                ) as ser:
                    ser.reset_input_buffer()
                    print("Streaming... Press Ctrl+C to stop.")

                    while not stop:
                        elapsed = time.monotonic() - start_time
                        if args.duration is not None and elapsed >= args.duration:
                            stop = True
                            break

                        raw = ser.readline()
                        if not raw:
                            continue

                        line = raw.decode("utf-8", errors="replace")
                        parsed = parse_sample_line(line)
                        if parsed is None:
                            continue

                        sample = Sample(elapsed, parsed[0], parsed[1], parsed[2], parsed[3])
                        sample_count += 1

                        if csv_writer:
                            csv_writer.write(sample)
                        if plotter:
                            plotter.add(sample)

            except serial.SerialException as exc:
                print(f"Serial disconnected/error: {exc}", file=sys.stderr)
                print(
                    f"Retrying in {args.reconnect_sec:.1f}s. "
                    "Close any open Serial Monitor/Plotter that may hold the port.",
                    file=sys.stderr,
                )
                time.sleep(args.reconnect_sec)
                continue

    finally:
        if csv_writer:
            csv_writer.close()
        if plotter:
            try:
                plotter.redraw()
                plotter.plt.ioff()
                plotter.plt.show(block=False)
                plotter.plt.pause(0.2)
            except Exception:
                pass

    print(f"Done. Samples captured: {sample_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
