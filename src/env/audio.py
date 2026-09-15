"""Records the game's own audio alongside a run.

libsm64 compiles in the decompilation's audio engine, so the samples here are the game's, driven
by the same ``play_sound`` calls Mario's actions make rather than by anything invented. That
matters for the backwards long jump specifically: ``act_long_jump`` plays ``SOUND_MARIO_YAHOO``,
and a working chain re-enters that action on nearly every frame, so the exploit has a sound and it
is Mario yelling without pause.

The engine fills two stereo buffers per call and picks their length from whether the caller is
starved, so asking as a starved caller every frame yields a constant sample count per frame.
Declaring the stream at that count times 30 makes its duration match the frame count exactly, with
no drift to reconcile against the replay.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import wave

from src.env.native import AUDIO_SAMPLE_RATE, Sm64

CHANNELS = 2
SAMPLE_WIDTH = 2


class AudioRecorder:
    """Accumulates one run's PCM and writes it out.

    Attributes:
        frames: Number of frames captured so far.
    """

    def __init__(self, game: Sm64, volume: float = 1.0) -> None:
        """Initializes the ROM's audio banks.

        Args:
            game: A live libsm64 handle. Its ROM buffer supplies the audio banks.
            volume: Master volume. One Mario wants the game's own 1.0, while a population sums
                into the same mixer and clips at full scale, so a swarm wants less.
        """
        game.audio_init()
        game.set_sound_volume(volume)
        self._game = game
        self._pcm = bytearray()
        self.frames = 0

    def capture(self) -> None:
        """Synthesizes and stores one frame of audio. Call once per ticked frame."""
        self._pcm += self._game.audio_tick()
        self.frames += 1

    @property
    def seconds(self) -> float:
        """Returns the duration of the captured audio."""
        return len(self._pcm) / (CHANNELS * SAMPLE_WIDTH * AUDIO_SAMPLE_RATE)

    @property
    def peak(self) -> int:
        """Returns the largest absolute sample value, or zero for silence."""
        if not self._pcm:
            return 0
        view = memoryview(self._pcm).cast("h")
        return max(max(view), -min(view))

    def wav_bytes(self) -> bytes:
        """Returns the capture as a RIFF WAV file."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(CHANNELS)
            handle.setsampwidth(SAMPLE_WIDTH)
            handle.setframerate(AUDIO_SAMPLE_RATE)
            handle.writeframes(bytes(self._pcm))
        return buffer.getvalue()

    def write(self, path: str, bitrate: str = "96k") -> str:
        """Writes the capture, encoding to mp3 when ffmpeg is available.

        A 37 second capture is about 4.8 MB as WAV and about 440 KB as mp3, and a viewer page has
        a 16 MB budget to share with geometry, so the compressed form is worth reaching for.
        Falls back to WAV rather than failing, and returns whichever path it actually wrote.

        Args:
            path: Desired output path. The extension is replaced to match what was written.
            bitrate: mp3 bitrate to ask ffmpeg for.

        Returns:
            The path written.

        Raises:
            ValueError: If nothing has been captured.
        """
        if not self._pcm:
            raise ValueError("no audio captured; call capture() once per frame")

        stem = os.path.splitext(path)[0]
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            out = f"{stem}.mp3"
            result = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "s16le",
                 "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(CHANNELS), "-i", "pipe:0",
                 "-c:a", "libmp3lame", "-b:a", bitrate, out],
                input=bytes(self._pcm), capture_output=True, check=False)
            if result.returncode == 0 and os.path.exists(out):
                return out

        out = f"{stem}.wav"
        with open(out, "wb") as handle:
            handle.write(self.wav_bytes())
        return out
