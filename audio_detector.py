import math
import threading
import time
import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


class AudioClapDetector:
    """Detects hand claps and double-claps using real-time microphone input."""

    def __init__(
        self,
        sample_rate=16000,
        block_size=512,
        peak_threshold=0.20,
        energy_ratio_threshold=2.8,
        min_clap_interval=0.12,
        max_clap_interval=1.0,
        action_cooldown=3.5,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.peak_threshold = peak_threshold
        self.energy_ratio_threshold = energy_ratio_threshold
        self.min_clap_interval = min_clap_interval
        self.max_clap_interval = max_clap_interval
        self.action_cooldown = action_cooldown

        self._stream = None
        self._lock = threading.Lock()
        self._running = False
        self._last_clap_time = float("-inf")
        self._last_trigger_time = float("-inf")
        self._double_clap_detected = False
        self._ambient_rms = 0.01
        self._status = "MIC: INACTIVO"

    def process_samples(self, samples, timestamp=None):
        """Processes a chunk of audio samples and updates clap state."""
        now = time.monotonic() if timestamp is None else timestamp
        if len(samples.shape) > 1:
            samples = samples[:, 0]

        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) > 0 else 0.0

        # Exponential moving average for background noise floor
        if rms < self._ambient_rms * 2.0 or self._ambient_rms <= 1e-6:
            self._ambient_rms = 0.95 * self._ambient_rms + 0.05 * max(rms, 1e-4)

        # Clap check: impulsive peak and elevated energy above ambient floor
        is_impulse = (
            peak >= self.peak_threshold
            and rms >= self._ambient_rms * self.energy_ratio_threshold
        )

        with self._lock:
            if is_impulse and (now - self._last_clap_time >= self.min_clap_interval):
                elapsed = now - self._last_clap_time
                if 0.15 <= elapsed <= self.max_clap_interval:
                    if now - self._last_trigger_time >= self.action_cooldown:
                        self._double_clap_detected = True
                        self._last_trigger_time = now
                        self._status = "MIC: 2 PALMADAS DETECTADAS!"
                    self._last_clap_time = float("-inf")
                else:
                    self._last_clap_time = now
                    self._status = "MIC: 1 palmada escuchada..."
            elif now - self._last_clap_time > self.max_clap_interval:
                if self._running:
                    self._status = "MIC ACTIVO (Escuchando 2 palmadas)"

    def poll_double_clap(self):
        """Returns True once when a double clap has been detected, resetting the flag."""
        with self._lock:
            detected = self._double_clap_detected
            self._double_clap_detected = False
            return detected

    @property
    def status(self):
        with self._lock:
            return self._status

    def start(self):
        if sd is None:
            self._status = "MIC ERROR: sounddevice no disponible"
            return False

        try:
            self._running = True
            self._status = "MIC ACTIVO (Escuchando 2 palmadas)"

            def audio_callback(indata, frames, time_info, status):
                self.process_samples(indata)

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                callback=audio_callback,
                dtype="float32",
            )
            self._stream.start()
            return True
        except Exception as error:
            self._running = False
            self._status = f"MIC INDISPONIBLE ({error})"
            self._stream = None
            return False

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._status = "MIC OFF"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
