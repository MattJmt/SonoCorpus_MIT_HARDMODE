#!/usr/bin/env python3
"""Realtime finger-hit inference from Arduino serial stream.

This script loads a trained checkpoint from train_hit_classifier.py and performs
sliding-window inference on EMA-filtered serial samples. It renders 4 live
colored squares in matplotlib for index/middle/ring/thumb activity.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import matplotlib.pyplot as plt
import numpy as np
import serial
import torch
from matplotlib.patches import Rectangle
from pythonosc.udp_client import SimpleUDPClient
from serial.tools import list_ports

from train_hit_classifier import FINGERS, HitConvNet

CHANNEL_NAMES = ("Index", "Middle", "Ring", "Thumb")
EXPECTED_HEADER = "\t".join(CHANNEL_NAMES)
DEFAULT_BAUD = 115200
DEFAULT_RECONNECT_SEC = 1.0
DEFAULT_EMA_ALPHA = 0.2
UDP_HOST = "127.0.0.1"
UDP_PORT = 6969
OSC_ADDRESS = "/glove"


@dataclass
class Sample:
    index: int
    middle: int
    ring: int
    thumb: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live finger-hit inference from serial stream")
    p.add_argument("--port", help="Serial device path (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/finger_hit_model.pt"),
        help="Model checkpoint path saved by train_hit_classifier.py",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (default: checkpoint meta threshold)",
    )
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="Override EMA alpha (default: checkpoint meta ema_alpha or 0.2)",
    )
    p.add_argument(
        "--reconnect-sec",
        type=float,
        default=DEFAULT_RECONNECT_SEC,
        help="Delay between reconnect attempts after serial disconnects",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional run duration in seconds",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Inference device selection",
    )
    p.add_argument("--udp-host", default=UDP_HOST, help="OSC UDP destination host.")
    p.add_argument("--udp-port", type=int, default=UDP_PORT, help="OSC UDP destination port.")
    p.add_argument("--osc-address", default=OSC_ADDRESS, help="OSC address path.")
    return p.parse_args()


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


def parse_sample_line(line: str) -> Optional[Sample]:
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
        values = [int(p) for p in parts]
    except ValueError:
        return None

    return Sample(values[0], values[1], values[2], values[3])


def ema_step(x: np.ndarray, prev: Optional[np.ndarray], alpha: float) -> np.ndarray:
    if prev is None:
        return x.astype(np.float32, copy=True)
    return (alpha * x + (1.0 - alpha) * prev).astype(np.float32, copy=False)


def choose_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FingerSquaresUI:
    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.fig.canvas.manager.set_window_title("SonoCorpus Live Finger Hits")
        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")

        self.on_colors = {
            "index": "#e74c3c",
            "middle": "#2ecc71",
            "ring": "#3498db",
            "thumb": "#f39c12",
        }
        self.off_color = "#d0d3d4"

        self.rects: list[Rectangle] = []
        self.labels = []
        for i, finger in enumerate(FINGERS):
            rect = Rectangle((i + 0.05, 0.2), 0.9, 0.6, facecolor=self.off_color, edgecolor="black")
            self.ax.add_patch(rect)
            self.rects.append(rect)

            label = self.ax.text(
                i + 0.5,
                0.1,
                finger,
                ha="center",
                va="center",
                fontsize=12,
                weight="bold",
            )
            self.labels.append(label)

        self.status_text = self.ax.text(2.0, 0.92, "Waiting for samples...", ha="center", va="center")
        self.prob_text = self.ax.text(2.0, 0.03, "", ha="center", va="center", fontsize=10)

        plt.ion()
        self.fig.tight_layout()
        self.fig.show()
        self._last_draw = 0.0
        self._draw_interval_s = 1.0 / 25.0

    def update(self, active: np.ndarray, probs: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_draw < self._draw_interval_s:
            return

        for i, finger in enumerate(FINGERS):
            color = self.on_colors[finger] if active[i] else self.off_color
            self.rects[i].set_facecolor(color)

        if np.any(active):
            active_names = [name for i, name in enumerate(FINGERS) if active[i]]
            self.status_text.set_text(f"HIT: {', '.join(active_names)}")
        else:
            self.status_text.set_text("NOISE / no hit")

        prob_line = "  ".join(f"{name}:{probs[i]:.2f}" for i, name in enumerate(FINGERS))
        self.prob_text.set_text(prob_line)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        self._last_draw = now


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[HitConvNet, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint or "meta" not in checkpoint:
        raise ValueError("Invalid checkpoint format: expected keys model_state_dict and meta")

    meta = checkpoint["meta"]
    model = HitConvNet(in_channels=4, out_dims=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, meta


def main() -> int:
    args = parse_args()

    if args.reconnect_sec <= 0:
        print("Error: --reconnect-sec must be > 0", file=sys.stderr)
        return 2
    if args.udp_port <= 0 or args.udp_port > 65535:
        print("Error: --udp-port must be in [1, 65535]", file=sys.stderr)
        return 2
    if not args.osc_address.startswith("/"):
        print("Error: --osc-address must start with '/'", file=sys.stderr)
        return 2

    device = choose_device(args.device)
    model, meta = load_model(args.checkpoint, device)

    window = int(meta.get("window", 16))
    mean = np.asarray(meta.get("mean", [0, 0, 0, 0]), dtype=np.float32)
    std = np.asarray(meta.get("std", [1, 1, 1, 1]), dtype=np.float32)
    std = np.maximum(std, 1e-6)
    threshold = float(args.threshold if args.threshold is not None else meta.get("threshold", 0.5))
    ema_alpha = float(args.ema_alpha if args.ema_alpha is not None else meta.get("ema_alpha", DEFAULT_EMA_ALPHA))
    if not (0.0 < ema_alpha <= 1.0):
        print("Error: --ema-alpha must be in (0, 1]", file=sys.stderr)
        return 2

    print(f"Loaded model: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Window: {window}, Threshold: {threshold:.3f}, EMA alpha: {ema_alpha:.3f}")
    print(f"OSC confidence stream: {args.udp_host}:{args.udp_port} {args.osc_address}")

    ui = FingerSquaresUI()
    osc_client = SimpleUDPClient(args.udp_host, args.udp_port)
    stop = False

    def _handle_sigint(_sig: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    fixed_port = args.port
    start_t = time.monotonic()
    buffer: Deque[np.ndarray] = deque(maxlen=window)
    ema_state: Optional[np.ndarray] = None
    last_announced_port: Optional[str] = None

    try:
        while not stop:
            elapsed = time.monotonic() - start_t
            if args.duration is not None and elapsed >= args.duration:
                break

            port = fixed_port or autodetect_port()
            if not port:
                print(f"No serial port found. Retrying in {args.reconnect_sec:.1f}s...", file=sys.stderr)
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
                    ema_state = None

                    while not stop:
                        elapsed = time.monotonic() - start_t
                        if args.duration is not None and elapsed >= args.duration:
                            stop = True
                            break

                        raw = ser.readline()
                        if not raw:
                            continue

                        line = raw.decode("utf-8", errors="replace")
                        sample = parse_sample_line(line)
                        if sample is None:
                            continue

                        x = np.asarray([sample.index, sample.middle, sample.ring, sample.thumb], dtype=np.float32)
                        x_ema = ema_step(x=x, prev=ema_state, alpha=ema_alpha)
                        ema_state = x_ema
                        buffer.append(x_ema)

                        if len(buffer) < window:
                            continue

                        window_arr = np.stack(buffer, axis=0)
                        window_arr = (window_arr - mean) / std
                        xt = torch.from_numpy(np.transpose(window_arr[None, :, :], (0, 2, 1))).float().to(device)

                        with torch.no_grad():
                            logits = model(xt)
                            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                        osc_values = (
                            round(float(probs[FINGERS.index("thumb")]), 2),
                            round(float(probs[FINGERS.index("index")]), 2),
                            round(float(probs[FINGERS.index("middle")]), 2),
                            round(float(probs[FINGERS.index("ring")]), 2),
                        )
                        osc_client.send_message(args.osc_address, list(osc_values))

                        active = probs >= threshold
                        ui.update(active=active, probs=probs)

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
        plt.close("all")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
