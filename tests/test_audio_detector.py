import unittest
import numpy as np

from audio_detector import AudioClapDetector


def make_clap(length=512, peak=0.7):
    t = np.arange(length)
    envelope = np.exp(-t / 30.0)
    samples = np.random.uniform(-1, 1, length) * envelope
    return (samples / np.max(np.abs(samples)) * peak).astype(np.float32)


def make_silence(length=512, noise=0.01):
    return np.random.uniform(-noise, noise, length).astype(np.float32)


class AudioClapDetectorTests(unittest.TestCase):
    def test_silence_does_not_trigger(self):
        detector = AudioClapDetector()
        silence = make_silence()
        for i in range(10):
            detector.process_samples(silence, timestamp=i * 0.05)
        self.assertFalse(detector.poll_double_clap())
        self.assertNotIn("palmada", detector.status.lower())

    def test_single_clap_does_not_trigger_double_clap(self):
        detector = AudioClapDetector()
        silence = make_silence()
        detector.process_samples(silence, timestamp=0.0)
        detector.process_samples(make_clap(), timestamp=0.1)
        self.assertFalse(detector.poll_double_clap())
        self.assertIn("1 palmada", detector.status)

    def test_double_clap_in_valid_window_triggers(self):
        detector = AudioClapDetector()
        detector.process_samples(make_silence(), timestamp=0.0)
        # Primer aplauso
        detector.process_samples(make_clap(), timestamp=0.2)
        self.assertFalse(detector.poll_double_clap())

        # Silencio intermedio
        detector.process_samples(make_silence(), timestamp=0.3)

        # Segundo aplauso a los 350ms del primero
        detector.process_samples(make_clap(), timestamp=0.55)
        self.assertTrue(detector.poll_double_clap())
        # Una vez polled, debe volver a False
        self.assertFalse(detector.poll_double_clap())

    def test_two_claps_too_far_apart_do_not_trigger(self):
        detector = AudioClapDetector(max_clap_interval=0.8)
        detector.process_samples(make_silence(), timestamp=0.0)
        detector.process_samples(make_clap(), timestamp=0.2)
        # Segundo aplauso 1.5s después (expiró)
        detector.process_samples(make_clap(), timestamp=1.7)
        self.assertFalse(detector.poll_double_clap())

    def test_reverberation_spike_is_debounced(self):
        detector = AudioClapDetector(min_clap_interval=0.12)
        detector.process_samples(make_silence(), timestamp=0.0)
        detector.process_samples(make_clap(), timestamp=0.2)
        # Pico de reverberación a los 40ms
        detector.process_samples(make_clap(), timestamp=0.24)
        self.assertFalse(detector.poll_double_clap())


if __name__ == "__main__":
    unittest.main()
