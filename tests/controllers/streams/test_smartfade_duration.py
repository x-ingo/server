"""Tests for reported track duration when (smart) crossfade holds back audio (issue #5494)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ContentType, MediaType
from music_assistant_models.errors import QueueEmpty
from music_assistant_models.media_items import AudioFormat

from music_assistant.controllers.streams.audio import StreamsAudio
from music_assistant.models.smart_fades import SmartFadesMode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# small mono format keeps the buffers tiny: 1 second == 16000 bytes
PCM_FORMAT = AudioFormat(
    content_type=ContentType.PCM_S16LE, sample_rate=8000, bit_depth=16, channels=1
)
SAMPLE_SIZE = PCM_FORMAT.pcm_sample_size


# --- flow mode ---


@pytest.mark.asyncio
async def test_flow_duration_preserved_when_seeking_near_end_mid_queue() -> None:
    """A seek near the end of a crossfaded track must not shrink its reported duration."""
    audio = _make_streams_audio()
    # 180s track, seeked to 135s: only 45s left to stream, of which the
    # crossfade hold-back is a large share - the regression scenario of #5494
    track_a = _make_queue_track("track-a", duration=180, seek_position=135.0)
    track_b = _make_queue_track("track-b", duration=60)
    _install_fake_item_stream(audio, {"track-a": 45, "track-b": 60})
    _install_fake_mixer(
        audio,
        fadein_trimmed_duration=0.0,
        crossfade_duration=10.0,
        pre_crossfade_duration=27.0,
    )
    audio.mass.player_queues.load_next_queue_item = AsyncMock(  # type: ignore[method-assign]
        side_effect=[track_b, QueueEmpty()]
    )
    audio.crossfade_allowed = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda queue_item, **_kwargs: queue_item is track_a
    )

    queue = _make_queue()
    bytes_yielded = 0
    async for chunk in audio.get_queue_flow_stream(queue, track_a, PCM_FORMAT):
        bytes_yielded += len(chunk)

    assert track_a.streamdetails.duration == 180
    assert track_a.duration == 180
    assert track_b.streamdetails.duration == 60
    assert track_b.duration == 60
    # 45s of A + 60s of B, minus the 10s crossfade overlap
    assert bytes_yielded == 95 * SAMPLE_SIZE


@pytest.mark.asyncio
async def test_flow_duration_preserved_on_last_track_of_queue() -> None:
    """The end-of-queue tail flush must not double-count or skip the held-back tail."""
    audio = _make_streams_audio()
    track_a = _make_queue_track("track-a", duration=180, seek_position=135.0)
    _install_fake_item_stream(audio, {"track-a": 45})
    audio.mass.player_queues.load_next_queue_item = AsyncMock(  # type: ignore[method-assign]
        side_effect=QueueEmpty()
    )
    audio.crossfade_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]

    queue = _make_queue()
    bytes_yielded = 0
    async for chunk in audio.get_queue_flow_stream(queue, track_a, PCM_FORMAT):
        bytes_yielded += len(chunk)

    assert track_a.streamdetails.duration == 180
    assert track_a.duration == 180
    assert track_a.streamdetails.seconds_streamed == 45
    # all 45 streamed seconds reach the player, tail included
    assert bytes_yielded == 45 * SAMPLE_SIZE


# --- single item (non-flow) mode ---


@pytest.mark.asyncio
async def test_smartfade_stream_duration_accounts_for_trimmed_tail() -> None:
    """The outro trimmed away by the smart crossfade still counts towards the duration."""
    audio = _make_streams_audio()
    track_a = _make_queue_track("track-a", duration=180)
    track_b = _make_queue_track("track-b", duration=60)
    _install_fake_item_stream(audio, {"track-a": 180, "track-b": 60})
    # 45s tail, 10s crossfade, 30s pre-crossfade: 5s of outro is trimmed away
    _install_fake_mixer(
        audio,
        fadein_trimmed_duration=0.0,
        crossfade_duration=10.0,
        pre_crossfade_duration=30.0,
    )
    audio.mass.player_queues.get = MagicMock(  # type: ignore[method-assign]
        return_value=_make_queue()
    )
    audio.mass.player_queues.load_next_queue_item = AsyncMock(  # type: ignore[method-assign]
        return_value=track_b
    )
    audio.select_pcm_format = AsyncMock(return_value=PCM_FORMAT)  # type: ignore[method-assign]
    audio.crossfade_allowed = MagicMock(return_value=True)  # type: ignore[method-assign]

    player = MagicMock()
    player.player_id = "player1"
    player.name = "Test Player"
    async for _chunk in audio.get_queue_item_stream_with_smartfade(player, track_a, PCM_FORMAT):
        pass

    assert track_a.streamdetails.duration == 180
    assert track_a.duration == 180


# --- helpers ---


def _make_streams_audio() -> StreamsAudio:
    """Return a StreamsAudio with smart crossfade enabled and all mass internals mocked."""
    audio = StreamsAudio(MagicMock())
    audio.mass.config.get_player_config_value = AsyncMock(  # type: ignore[method-assign]
        return_value=SmartFadesMode.SMART_CROSSFADE
    )
    audio.mass.config.get_raw_player_config_value = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda _player_id, _key, default=None: default
    )
    audio.mass.config.get_raw_core_config_value = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda _domain, _key, default=None: default
    )
    audio.mass.players.get_player = MagicMock(return_value=None)  # type: ignore[method-assign]
    audio._flow_stream_needs_restart = MagicMock(return_value=False)  # type: ignore[method-assign]
    return audio


def _make_queue() -> MagicMock:
    """Build a queue double with the fields the stream generators touch."""
    queue = MagicMock()
    queue.queue_id = "queue1"
    queue.display_name = "Test Queue"
    queue.session_id = "session1"
    return queue


def _make_queue_track(item_id: str, *, duration: int, seek_position: float = 0.0) -> MagicMock:
    """Build a queue item double with real numbers on the fields used for bookkeeping."""
    track = MagicMock()
    track.queue_item_id = item_id
    track.queue_id = "queue1"
    track.name = item_id
    track.media_type = MediaType.TRACK
    track.duration = duration
    track.extra_attributes = {}
    track.streamdetails = MagicMock()
    track.streamdetails.uri = f"file://{item_id}"
    track.streamdetails.duration = duration
    track.streamdetails.seek_position = seek_position
    track.streamdetails.seconds_streamed = None
    return track


def _install_fake_item_stream(audio: StreamsAudio, seconds_by_item: dict[str, int]) -> None:
    """Replace get_queue_item_stream with one yielding 1-second silence chunks per item."""

    async def _fake_stream(
        queue_item: MagicMock, pcm_format: AudioFormat = PCM_FORMAT, **_kwargs: Any
    ) -> AsyncGenerator[bytes]:
        for _ in range(seconds_by_item[queue_item.queue_item_id]):
            yield b"\x00" * pcm_format.pcm_sample_size

    audio.get_queue_item_stream = _fake_stream  # type: ignore[method-assign, assignment]


def _install_fake_mixer(
    audio: StreamsAudio,
    *,
    fadein_trimmed_duration: float,
    crossfade_duration: float,
    pre_crossfade_duration: float,
) -> None:
    """Replace the smart fades mixer with one producing output of the timing-implied length."""
    timing_info = MagicMock()
    timing_info.fadein_trimmed_duration = fadein_trimmed_duration
    timing_info.crossfade_duration = crossfade_duration
    timing_info.pre_crossfade_duration = pre_crossfade_duration
    smart_fade = MagicMock()
    smart_fade.timing_info = timing_info

    async def _mix(
        _smart_fade: MagicMock,
        *,
        fade_in_part: bytes | AsyncGenerator[bytes],
        fade_out_part: bytes,  # noqa: ARG001  # keyword name must match the real mixer
        pcm_format: AudioFormat,
    ) -> AsyncGenerator[bytes]:
        fade_in = bytearray()
        if isinstance(fade_in_part, bytes):
            fade_in.extend(fade_in_part)
        else:
            async for chunk in fade_in_part:
                fade_in.extend(chunk)
        # mix output = PRE + CF + POST, where the fade-out trim and the
        # crossfade overlap shorten the combined inputs
        out_seconds = (
            pre_crossfade_duration
            + crossfade_duration
            + len(fade_in) / pcm_format.pcm_sample_size
            - fadein_trimmed_duration
            - crossfade_duration
        )
        output = b"\x00" * int(out_seconds * pcm_format.pcm_sample_size)
        for offset in range(0, len(output), pcm_format.pcm_sample_size):
            yield output[offset : offset + pcm_format.pcm_sample_size]

    mixer = MagicMock()
    mixer.build = AsyncMock(return_value=smart_fade)
    mixer.mix = _mix
    audio._smart_fades_mixer = mixer
