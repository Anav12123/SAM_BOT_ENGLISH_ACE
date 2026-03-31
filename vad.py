"""
vad.py — Silero Voice Activity Detection via ONNX Runtime

Lightweight (~50MB onnxruntime + ~2MB model). No PyTorch needed.
Processes 32ms audio chunks and returns speech probability 0.0-1.0.

Usage:
    vad = SileroVAD()
    await vad.setup()  # downloads model on first run
    confidences = vad.process_chunk(pcm_bytes)  # list of floats
"""

import os
import time
import numpy as np


class SileroVAD:
    ONNX_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx")
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512  # 32ms at 16kHz — required by Silero

    def __init__(self):
        self._session = None
        self._state = None
        self._ready = False
        self._audio_buffer = np.array([], dtype=np.float32)

        # State tracking
        self.is_speaking = False
        self.speech_start = 0.0
        self.silence_start = 0.0
        self.last_confidence = 0.0

    async def setup(self):
        """Download ONNX model + initialize session. Call once at startup."""
        try:
            if not os.path.exists(self.MODEL_PATH):
                print("[VAD] Downloading Silero VAD ONNX model...")
                import httpx
                async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                    resp = await client.get(self.ONNX_URL)
                    resp.raise_for_status()
                    with open(self.MODEL_PATH, "wb") as f:
                        f.write(resp.content)
                    print(f"[VAD] Model downloaded ({len(resp.content) // 1024}KB)")
            else:
                print("[VAD] ONNX model found on disk")

            import onnxruntime
            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 3  # suppress warnings
            self._session = onnxruntime.InferenceSession(
                self.MODEL_PATH, sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self._reset_state()

            # Test inference to verify model works
            test_audio = np.zeros(self.CHUNK_SAMPLES, dtype=np.float32)
            test_conf = self._infer(test_audio)
            self._reset_state()  # reset after test
            print(f"[VAD] Test inference: conf={test_conf:.4f} (expected ~0 for silence)")

            self._ready = True
            print("[VAD] ✅ Silero VAD ready (ONNX, ~1ms/chunk)")
        except Exception as e:
            print(f"[VAD] ⚠️  Setup failed: {e} — VAD disabled, using speech_off fallback")
            self._ready = False

    # Recall.ai mixed audio is much quieter than direct microphone
    # Amplify before feeding to Silero which was trained on direct mic input
    AUDIO_GAIN = 100.0

    def _reset_state(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._audio_buffer = np.array([], dtype=np.float32)

    def process_chunk(self, pcm_bytes: bytes) -> list[float]:
        """Feed raw PCM bytes (16kHz S16LE mono). Returns list of confidence values."""
        if not self._ready:
            return []

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Amplify quiet meeting audio — clip to [-1, 1] to prevent distortion
        samples = np.clip(samples * self.AUDIO_GAIN, -1.0, 1.0)

        self._audio_buffer = np.concatenate([self._audio_buffer, samples])

        confidences = []
        while len(self._audio_buffer) >= self.CHUNK_SAMPLES:
            chunk = self._audio_buffer[:self.CHUNK_SAMPLES]
            self._audio_buffer = self._audio_buffer[self.CHUNK_SAMPLES:]
            conf = self._infer(chunk)
            confidences.append(conf)

        return confidences

    def _infer(self, chunk: np.ndarray) -> float:
        """Run one 32ms window. Returns speech probability 0.0-1.0."""
        try:
            ort_inputs = {
                "input": chunk.reshape(1, -1),
                "state": self._state,
                "sr": np.array(self.SAMPLE_RATE, dtype=np.int64),
            }
            ort_outs = self._session.run(None, ort_inputs)
            self._state = ort_outs[1]
            return float(ort_outs[0].item())
        except Exception as e:
            if not hasattr(self, '_infer_error_logged'):
                print(f"[VAD] ⚠️  Inference error: {e}")
                print(f"[VAD] Model inputs: {[i.name for i in self._session.get_inputs()]}")
                print(f"[VAD] Model input shapes: {[(i.name, i.shape, i.type) for i in self._session.get_inputs()]}")
                self._infer_error_logged = True
            return 0.0

    def update_state(self, confidence: float, threshold: float = 0.5):
        """Update speaking/silence tracking from a single confidence value."""
        self.last_confidence = confidence
        now = time.time()

        if confidence >= threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start = now
                print(f"[VAD] 🎙️ Speech started (conf={confidence:.2f})")
            self.silence_start = 0.0
        else:
            if self.is_speaking and self.silence_start == 0.0:
                self.silence_start = now
            # Auto-reset if silence exceeds 3s — stale speech detection
            elif self.is_speaking and self.silence_start > 0.0:
                if (now - self.silence_start) > 3.0:
                    self.is_speaking = False
                    self.silence_start = 0.0

    def silence_duration_ms(self) -> float:
        """Milliseconds of continuous silence. 0 if still speaking or no silence yet."""
        if not self.is_speaking or self.silence_start == 0.0:
            return 0.0
        return (time.time() - self.silence_start) * 1000

    def end_turn(self):
        """Mark turn as finished. Call after flushing buffer."""
        self.is_speaking = False
        self.silence_start = 0.0
        self.speech_start = 0.0

    def reset(self):
        """Full reset — call between meetings."""
        self._reset_state()
        self.end_turn()
        self.last_confidence = 0.0

    @property
    def ready(self) -> bool:
        return self._ready
