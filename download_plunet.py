#!/usr/bin/env python3
"""Download source media from Plunet with readable terminal progress."""

import sys
import time
from pathlib import Path

from config import AUDIO_EXTENSIONS, load_config
from plunet_order_client import PlunetOrderClient

# Smaller files first. expected_bytes from prior Plunet probes (None = unknown).
DOWNLOAD_QUEUE = [
    ("O-33603", r"\source\es-mx\Recordings-0000000248.wav", 76_341_298),
    ("O-33603", r"\source\es-mx\Recordings-0000000246.wav", 16_760_882),
    ("O-33603", r"\source\es-mx\Recordings-0000000249.wav", None),
    ("O-33687", r"\source\en-us\Axon_Interview-Internal_Affairs-Interrogation_Room-Dome_Cam.mp4", 254_000_000),
    ("O-33693", r"\source\en-us\TFC_Messick_Interview July 29 2025.mp4", 171_000_000),
]

OUTPUT_DIR = Path(__file__).parent / "downloads"
PROGRESS_INTERVAL_SEC = 3.0


def _fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024 / 1024:.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(max(0, seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _render_bar(pct: float, width: int = 28) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class TerminalProgress:
    def __init__(self, label: str, expected_bytes: int | None = None):
        self.label = label
        self.expected_bytes = expected_bytes
        self.started = time.time()
        self.last_print = 0.0
        self.last_received = 0

    def _eta(self, received: int, est_total: int, speed: float) -> str:
        if speed <= 0 or est_total <= received:
            return "?"
        return _fmt_time((est_total - received) / speed)

    def update(self, received: int, total: int | None):
        now = time.time()
        if now - self.last_print < PROGRESS_INTERVAL_SEC and received > 0:
            return

        elapsed = now - self.started
        if total is not None:
            try:
                total = int(total)
            except (TypeError, ValueError):
                total = None
        est_total = total or self.expected_bytes
        speed = received / elapsed if elapsed > 0 else 0

        if est_total and est_total > 0:
            pct = received * 100 / est_total
            line = (
                f"{self.label}\n"
                f"  {_render_bar(pct)} {pct:5.1f}%  "
                f"{_fmt_bytes(received)} / ~{_fmt_bytes(est_total)}\n"
                f"  speed {_fmt_bytes(int(speed))}/s  "
                f"elapsed {_fmt_time(elapsed)}  eta {self._eta(received, est_total, speed)}"
            )
        else:
            line = (
                f"{self.label}\n"
                f"  {_fmt_bytes(received)} received  "
                f"speed {_fmt_bytes(int(speed))}/s  "
                f"elapsed {_fmt_time(elapsed)}  "
                f"(total size unknown until Plunet finishes)"
            )

        print(line, flush=True)
        self.last_print = now
        self.last_received = received

    def done(self, nbytes: int, out_path: Path):
        elapsed = time.time() - self.started
        print(
            f"{self.label}\n"
            f"  DONE  {_render_bar(100)} 100.0%  "
            f"{_fmt_bytes(nbytes)} -> {out_path}\n"
            f"  time {_fmt_time(elapsed)}\n",
            flush=True,
        )

    def fail(self, err: str):
        print(f"{self.label}\n  FAILED: {err}\n", flush=True)


def _print_queue_plan(queue: list, output_dir: Path) -> None:
    print("Queue:")
    for idx, (order_name, file_path, expected_size) in enumerate(queue, 1):
        name = Path(file_path.replace("\\", "/")).name
        out_path = output_dir / order_name / name
        if out_path.exists() and out_path.stat().st_size > 0:
            status = f"skip (exists, {_fmt_bytes(out_path.stat().st_size)})"
        else:
            size_hint = f"~{_fmt_bytes(expected_size)}" if expected_size else "size unknown"
            status = f"download ({size_hint})"
        print(f"  {idx}. {order_name} / {name}  —  {status}")
    print()


def main():
    cfg = load_config()["plunet"]
    if not all(cfg.values()):
        print("Missing PLUNET_* credentials in .env", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_files = len(DOWNLOAD_QUEUE)
    ok = 0
    failed = 0
    skipped = 0

    print("=" * 60)
    print(f"Plunet download -> {OUTPUT_DIR.resolve()}")
    print(f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    _print_queue_plan(DOWNLOAD_QUEUE, OUTPUT_DIR)

    client = PlunetOrderClient(cfg["base_url"], cfg["username"], cfg["password"])
    tok = client.login()
    if not tok or tok == "refused":
        print(f"Plunet login failed: {tok!r}", file=sys.stderr)
        sys.exit(1)
    print("Plunet login OK\n")

    for idx, (order_name, file_path, expected_size) in enumerate(DOWNLOAD_QUEUE, 1):
        name = Path(file_path.replace("\\", "/")).name
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            print(f"[{idx}/{total_files}] skip (not media): {order_name} / {name}")
            continue

        order_dir = OUTPUT_DIR / order_name
        order_dir.mkdir(parents=True, exist_ok=True)
        out_path = order_dir / name

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{idx}/{total_files}] SKIP  {order_name} / {name}  ({_fmt_bytes(out_path.stat().st_size)})")
            skipped += 1
            ok += 1
            continue

        label = f"[{idx}/{total_files}] {order_name} / {name}"
        print("-" * 60)
        print(label)
        print("-" * 60)

        progress = TerminalProgress(label, expected_size)
        client.set_progress_callback(progress.update)

        try:
            order_id = client.order_name_to_id(order_name)
            filename, content = client.download_file(order_id, file_path, folder_type="source")
            out_path.write_bytes(content)
            progress.done(len(content), out_path)
            ok += 1
        except Exception as e:
            progress.fail(str(e))
            failed += 1
            print(f"  error: {e}", file=sys.stderr)

    print("=" * 60)
    print(f"Summary: {ok - skipped} downloaded, {skipped} skipped, {failed} failed, {total_files} total")
    print(f"Folder: {OUTPUT_DIR.resolve()}")
    print(f"Finished {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
