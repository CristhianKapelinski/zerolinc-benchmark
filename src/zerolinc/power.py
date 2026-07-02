"""GPU power sampling during a run (nvidia-smi at 1 Hz), for energy cost figures."""

import subprocess
import threading
import time


class PowerSampler:
    """Context manager sampling GPU power draw; yields mean/max watts and Wh."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.elapsed_s = 0.0

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                value = out.stdout.strip().splitlines()[0]
                self.samples.append(float(value))
            except (subprocess.SubprocessError, ValueError, IndexError, OSError):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PowerSampler":
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_s = time.perf_counter() - self._start
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def report(self) -> dict:
        if not self.samples:
            return {"gpu_power_mean_w": None, "gpu_power_max_w": None, "gpu_energy_wh": None}
        mean_w = sum(self.samples) / len(self.samples)
        return {
            "gpu_power_mean_w": round(mean_w, 1),
            "gpu_power_max_w": round(max(self.samples), 1),
            "gpu_energy_wh": round(mean_w * self.elapsed_s / 3600, 3),
        }
