from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig


class SpeechUnavailableError(RuntimeError):
    ...


@dataclass
class SpeechService:
    settings: AppConfig
    _whisper_model: object | None = None
    _tts_model: object | None = None

    async def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        if not self.settings.enable_speech_stt:
            raise SpeechUnavailableError("Speech-to-text disabled. Set NOTEBOOKLM_ENABLE_SPEECH_STT=1 to enable.")
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SpeechUnavailableError(
                "faster-whisper is required for STT. Install with pip install faster-whisper."
            ) from exc

        if self._whisper_model is None:
            model_dir = self.settings.models_dir / "whisper"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._whisper_model = WhisperModel("base.en", device="auto", compute_type="int8_float16")

        segments, _info = self._whisper_model.transcribe(str(audio_path), language=language, beam_size=1)
        transcript = " ".join(segment.text.strip() for segment in segments)
        return transcript.strip()

    async def synthesize(self, text: str) -> Path:
        if not self.settings.enable_speech_tts:
            raise SpeechUnavailableError("Text-to-speech disabled. Set NOTEBOOKLM_ENABLE_SPEECH_TTS=1 to enable.")
        try:
            import piper  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise SpeechUnavailableError("piper-tts is required for TTS. Install with pip install piper-tts.") from exc

        model_path = self.settings.models_dir / "piper" / "en_US-amy-medium.onnx"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            raise SpeechUnavailableError(
                f"TTS model missing: {model_path}. "
                "Download a Piper voice model and place it there."
            )

        synthesizer = piper.load(model_path, config_path=None)
        audio = synthesizer(text)
        import numpy as np  # type: ignore
        output_path = self.settings.cache_dir / f"tts_{abs(hash(text))}.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(np.asarray(audio).astype("int16").tobytes())
        return output_path
