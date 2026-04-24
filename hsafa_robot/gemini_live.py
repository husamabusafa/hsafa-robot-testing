"""gemini_live.py - Gemini Live API session wrapper.
 
Runs a Gemini Live session on a dedicated background thread so the main
control loop never blocks on network I/O. The session streams:
 
    * microphone (via ``mic_source``)       -> Gemini (PCM 16 kHz int16 mono)
    * latest camera JPEG, ~1 FPS            -> Gemini
    * Gemini's audio replies (via
      ``speaker_sink``)                     -> audio output (float32 mono 16 kHz)
 
Audio I/O is delegated to caller-provided callables so this module is
agnostic about the underlying backend. ``main.py`` plugs in Reachy's
GStreamer-based ``MediaManager`` (``reachy.media.get_audio_sample`` and
``reachy.media.push_audio_sample``), which handles device selection,
channel duplication, and device-rate resampling for us.
 
While Gemini is producing audio, the ``is_speaking`` :class:`threading.Event`
is set so other components (e.g. animation logic) can react.
"""
from __future__ import annotations
 
import asyncio
import logging
import math
import threading
import time
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional
 
import numpy as np
from google import genai
from google.genai import types
 
try:
    from scipy.signal import resample_poly  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    resample_poly = None  # type: ignore
    _HAVE_SCIPY = False
 
log = logging.getLogger(__name__)
 
 
# --- Audio constants -------------------------------------------------------
# Gemini Live expects 16 kHz PCM16 mono input and streams 24 kHz PCM16 mono
# output. Reachy's MediaManager expects 16 kHz float32, so we resample
# 24 -> 16 kHz ourselves (polyphase up=2 down=3).
GEMINI_INPUT_SR = 16000
GEMINI_OUTPUT_SR = 24000
SINK_SR = 16000
INT16_SCALE = 32768.0
 
 
# --- Default model & voice -------------------------------------------------
# Override via the ``GEMINI_MODEL`` env var or ``--model`` CLI flag if this
# model is ever retired.
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Puck"   # other options: Charon, Kore, Fenrir, Aoede
DEFAULT_VIDEO_FPS = 1.0  # Gemini Live processes video at ~1 FPS anyway
 
 
# Callable types handed in by ``main.py``.
FrameSource = Callable[[], Optional[bytes]]
# Returns a float32 ndarray shaped ``(N,)`` mono OR ``(N, C)`` multi-channel
# at 16 kHz, OR ``None`` if no data is available yet (non-blocking).
MicSource = Callable[[], Optional[np.ndarray]]
# Accepts a float32 ndarray of shape ``(N,)`` mono at 16 kHz.
SpeakerSink = Callable[[np.ndarray], None]
# Async tool handler: given the tool name + args, returns the JSON-
# serializable response dict that Gemini will see as the tool's output.
ToolHandler = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]
 
 
# How often to emit a one-line audio health summary (seconds).
DIAG_INTERVAL = 3.0
 
 
def _dbfs(value: float) -> float:
    """Convert a linear int16 magnitude (0..32768) to dBFS. -inf -> -99."""
    if value <= 0:
        return -99.0
    return 20.0 * math.log10(value / 32768.0)
 
 
class GeminiLiveSession:
    """Asynchronous Gemini Live session exposed to a synchronous main loop.
 
    Call :py:meth:`start` to launch the session; :py:meth:`stop` to tear it
    down. Read :py:attr:`is_speaking` each control tick to drive animation
    selection.
 
    Audio I/O is injected via two callables so the module stays agnostic
    about the underlying device:
 
    - ``mic_source()`` returns the latest captured audio chunk at 16 kHz
      (``(N,)`` mono or ``(N, C)`` multi-channel float32), or ``None`` if no
      new data is available yet.
    - ``speaker_sink(samples)`` accepts mono float32 at 16 kHz and is
      expected to be non-blocking (e.g. a GStreamer ``push_buffer`` call).
    """
 
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        voice_name: str = DEFAULT_VOICE,
        system_instruction: Optional[str] = None,
        frame_source: Optional[FrameSource] = None,
        video_fps: float = DEFAULT_VIDEO_FPS,
        mic_source: Optional[MicSource] = None,
        speaker_sink: Optional[SpeakerSink] = None,
        mic_poll_interval: float = 0.02,  # 20 ms -> ~320 samples @ 16 kHz
        mic_gate_tail_s: float = 0.6,
        barge_in_rms: float = 0.03,
        barge_in_min_chunks: int = 3,
        barge_in_hold_s: float = 2.0,
        tools: Optional[List[Any]] = None,
        tool_handler: Optional[ToolHandler] = None,
    ) -> None:
        if not api_key:
            raise ValueError("GeminiLiveSession requires a GEMINI_API_KEY")
        if not _HAVE_SCIPY:
            raise RuntimeError(
                "scipy is required for Gemini 24 kHz -> 16 kHz resampling. "
                "Install it with `pip install scipy`."
            )
 
        self._api_key = api_key
        self._model = model
        self._voice_name = voice_name
        self._system_instruction = system_instruction
        self._frame_source = frame_source
        self._video_period = 1.0 / max(video_fps, 0.01)
        self._mic_source = mic_source
        self._speaker_sink = speaker_sink
        self._mic_poll_interval = mic_poll_interval
        self._mic_gate_tail_s = mic_gate_tail_s
        # Optional Gemini function-calling. ``tools`` is a list of
        # ``types.Tool`` (each wraps one or more ``FunctionDeclaration``).
        # ``tool_handler`` is an async callable that receives
        # ``(name, args)`` and returns the JSON-serializable response.
        self._tools = list(tools) if tools else None
        self._tool_handler = tool_handler
        # Tracks in-flight tool-call tasks so they don't get GC'd and so
        # we can cancel them cleanly on teardown.
        self._tool_tasks: set[asyncio.Task] = set()
        # Barge-in threshold, measured as linear RMS (sqrt(mean(arr**2)))
        # of the downmixed float32 mono chunk.
        #
        # Why RMS and not peak: when Reachy's speaker plays Gemini's voice,
        # the mic picks up the bleed. Measured on this hardware the bleed
        # peak sits at roughly -25 dBFS (0.05 linear) but its RMS is around
        # -50 dBFS (0.003 linear). Human speech at normal volume has RMS
        # ~-25 to -30 dBFS (0.03-0.06). RMS therefore discriminates voice
        # from bleed by ~20 dB, while peak would false-trigger on the
        # first speaker transient and permanently arm the barge-in window,
        # which is exactly what was stopping Gemini from firing
        # ``server_content.interrupted``.
        self._barge_in_rms = float(barge_in_rms)
        # Require this many CONSECUTIVE chunks above the RMS threshold
        # before arming the barge-in hold. Rejects brief transients that
        # would otherwise keep poisoning the stream sent to Gemini's VAD.
        self._barge_in_min_chunks = max(1, int(barge_in_min_chunks))
        # Once sustained loud audio trips the threshold, keep the gate
        # OPEN for this many seconds so Gemini's server-side VAD receives
        # a continuous audio stream (it won't fire ``interrupted`` on
        # sparse loud bursts separated by gated silence).
        self._barge_in_hold_s = float(barge_in_hold_s)
        # Monotonic timestamp. Mic frames captured before this time are
        # dropped (not sent to Gemini) to avoid feeding the robot's own
        # speaker output back into Gemini's VAD, which would keep any
        # turn perpetually "open" and stall the conversation.
        self._mic_gate_until: float = 0.0
        # Monotonic timestamp. While now < _barge_in_until, ALL mic
        # frames are sent regardless of ``is_speaking``/``mic_gate_until``.
        self._barge_in_until: float = 0.0
        # Counter of consecutive loud chunks (resets on any quiet chunk).
        self._barge_in_run: int = 0
 
        # Cross-thread state
        self.is_speaking = threading.Event()
        self.connected = threading.Event()
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
 
        # Gemini Live caps each connection at ~10 minutes; passing this
        # handle on reconnect continues the same conversation so the robot
        # can run indefinitely without losing context.
        self._resumption_handle: Optional[str] = None
 
    # ---- public API -----------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Session already started")
        self._thread = threading.Thread(
            target=self._thread_main, name="gemini-live", daemon=True,
        )
        self._thread.start()
 
    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        # The background thread may have crashed (e.g. bad model name) and
        # closed its event loop already - guard against that.
        loop = self._loop
        if loop is not None:
            try:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
 
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
 
    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        return self.connected.wait(timeout=timeout)
 
    # ---- thread entry ---------------------------------------------------
    def _thread_main(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_session())
        except Exception as e:
            self._last_error = f"{e}\n{traceback.format_exc()}"
            log.error("Gemini session crashed: %s", self._last_error)
        finally:
            self.is_speaking.clear()
            self.connected.clear()
            try:
                if self._loop is not None:
                    self._loop.close()
            except Exception as e:
                log.debug("loop close: %s", e)
 
    # ---- session --------------------------------------------------------
    async def _run_session(self) -> None:
        """Outer loop: keep the session alive across server-side disconnects.
 
        Gemini Live drops any single websocket after ~10 minutes (or sooner
        if ``go_away`` is sent). We reconnect indefinitely, passing the
        latest session-resumption handle so the conversation continues
        seamlessly.
        """
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
 
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._run_one_connection(client)
                backoff = 1.0  # clean exit -> reset backoff
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._stop.is_set():
                    break
                self.is_speaking.clear()
                self.connected.clear()
                log.warning(
                    "Gemini Live connection dropped: %s; reconnecting in %.1fs",
                    e, backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2.0, 15.0)
 
    async def _run_one_connection(self, client: "genai.Client") -> None:
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._voice_name,
                    ),
                ),
            ),
            # Aggressive VAD so Gemini replies quickly after you stop
            # talking. Default ``silence_duration_ms`` is ~500 ms;
            # halving it trims ~250 ms off end-of-speech latency at the
            # cost of a slightly higher chance of premature cut-offs.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=(
                        types.StartSensitivity.START_SENSITIVITY_HIGH
                    ),
                    end_of_speech_sensitivity=(
                        types.EndSensitivity.END_SENSITIVITY_HIGH
                    ),
                    prefix_padding_ms=20,
                    silence_duration_ms=250,
                ),
                # Without this, some preview models default to
                # ``NO_INTERRUPTION`` and keep talking even after the
                # user's voice has been detected during the model's turn.
                activity_handling=(
                    types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS
                ),
            ),
            session_resumption=types.SessionResumptionConfig(
                handle=self._resumption_handle,
            ),
        )
        if self._tools:
            config.tools = self._tools
        if self._system_instruction:
            config.system_instruction = types.Content(
                parts=[types.Part(text=self._system_instruction)],
            )
 
        log.info(
            "Gemini Live: connecting to %s (resume=%s) ...",
            self._model, "yes" if self._resumption_handle else "no",
        )
        async with client.aio.live.connect(model=self._model, config=config) as session:
            self.connected.set()
            log.info("Gemini Live: connected")
            await asyncio.gather(
                self._mic_task(session),
                self._video_task(session),
                self._receive_task(session),
            )
 
    # ---- mic -> Gemini --------------------------------------------------
    async def _mic_task(self, session) -> None:
        """Poll ``mic_source`` and stream 16 kHz mono int16 PCM to Gemini."""
        if self._mic_source is None:
            log.info("mic: no mic_source provided, skipping input stream")
            return
 
        stats = {
            "sent": 0, "send_fail": 0,
            "peak": 0, "rms_sum_sq": 0.0, "blocks": 0,
            "empty_polls": 0, "gated": 0, "barge_in": 0,
            "gated_rms": 0.0,  # max linear RMS among GATED chunks
        }
 
        async def diag_loop():
            last = time.monotonic()
            while not self._stop.is_set():
                try:
                    await asyncio.sleep(DIAG_INTERVAL)
                except asyncio.CancelledError:
                    return
                now = time.monotonic()
                dt = max(now - last, 1e-3)
                last = now
                blocks = stats["blocks"]; stats["blocks"] = 0
                rms_sum_sq = stats["rms_sum_sq"]; stats["rms_sum_sq"] = 0.0
                peak = stats["peak"]; stats["peak"] = 0
                sent = stats["sent"]; stats["sent"] = 0
                empty = stats["empty_polls"]; stats["empty_polls"] = 0
                send_fail = stats["send_fail"]; stats["send_fail"] = 0
                gated = stats["gated"]; stats["gated"] = 0
                barge_in = stats["barge_in"]; stats["barge_in"] = 0
                grm_lin = stats["gated_rms"]; stats["gated_rms"] = 0.0
                gated_rms_i16 = int(grm_lin * 32768.0)
                rms = math.sqrt(rms_sum_sq / blocks) if blocks else 0.0
                # Typical speaking voice 30 cm from the mic: peak -6..-20
                # dBFS, RMS -25..-40 dBFS. Peak < -45 dBFS => Gemini will
                # likely miss you; peak at 0 dBFS => clipping.
                # ``gated_rms`` shows the loudest RMS among GATED chunks
                # (i.e. speaker bleed). Set ``barge_in_rms`` 6-10 dB above
                # this so real speech trips barge-in but bleed doesn't.
                log.info(
                    "MIC  sent=%d/%.1fs peak=%+.1fdBFS rms=%+.1fdBFS "
                    "blocks=%d empty=%d gated=%d (rms=%+.1fdBFS) "
                    "barge_in=%d send_fail=%d",
                    sent, dt, _dbfs(peak), _dbfs(rms),
                    blocks, empty, gated, _dbfs(gated_rms_i16),
                    barge_in, send_fail,
                )
 
        diag_task = asyncio.create_task(diag_loop(), name="mic-diag")
        try:
            while not self._stop.is_set():
                try:
                    sample = await asyncio.to_thread(self._mic_source)
                except Exception as e:
                    log.warning("mic_source raised: %s", e)
                    sample = None
 
                if sample is None or getattr(sample, "size", 0) == 0:
                    stats["empty_polls"] += 1
                    await asyncio.sleep(self._mic_poll_interval)
                    continue
 
                # Downmix and convert: accept (N,), (N, C), or (C, N).
                arr = np.asarray(sample, dtype=np.float32)
                if arr.ndim == 2:
                    # Prefer the "samples-first" axis; Reachy's GStreamer
                    # backend gives (N, 2) so this is a no-op for it.
                    if arr.shape[0] < arr.shape[1]:
                        arr = arr.T
                    arr = arr.mean(axis=1)
                np.clip(arr, -1.0, 1.0, out=arr)

                # Echo-gate: drop mic frames while Gemini is talking, plus
                # a short tail to let the speaker buffer and room echo
                # decay. Without this, the open mic feeds Gemini's VAD
                # the robot's own voice and turn_complete is never fired
                # -> conversation stalls after the first reply.
                # Exception: if the chunk's RMS is sustained above the
                # speaker-bleed level for multiple consecutive chunks,
                # it's the user talking over Gemini. We open a short
                # "barge-in hold" window during which ALL mic frames are
                # sent so Gemini's server VAD sees a continuous audio
                # stream and fires ``interrupted`` (sparse loud bursts
                # alone are not enough, and peak-based detection
                # false-triggers on speaker transients).
                now = time.monotonic()
                if arr.size:
                    chunk_rms = float(np.sqrt(np.mean(arr * arr)))
                else:
                    chunk_rms = 0.0
                if chunk_rms >= self._barge_in_rms:
                    self._barge_in_run += 1
                    if self._barge_in_run >= self._barge_in_min_chunks:
                        self._barge_in_until = now + self._barge_in_hold_s
                else:
                    self._barge_in_run = 0
                barge_in_active = now < self._barge_in_until
                gating = (
                    self.is_speaking.is_set()
                    or now < self._mic_gate_until
                )
                if gating and not barge_in_active:
                    stats["gated"] += 1
                    if chunk_rms > stats["gated_rms"]:
                        stats["gated_rms"] = chunk_rms
                    await asyncio.sleep(self._mic_poll_interval)
                    continue
                if barge_in_active and gating:
                    stats["barge_in"] += 1

                pcm16 = (arr * 32767.0).astype(np.int16)
 
                if pcm16.size:
                    peak = int(np.max(np.abs(pcm16)))
                    if peak > stats["peak"]:
                        stats["peak"] = peak
                    stats["rms_sum_sq"] += float(
                        np.mean(pcm16.astype(np.float32) ** 2)
                    )
                    stats["blocks"] += 1
 
                try:
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=pcm16.tobytes(),
                            mime_type=f"audio/pcm;rate={GEMINI_INPUT_SR}",
                        ),
                    )
                    stats["sent"] += 1
                except Exception as e:
                    stats["send_fail"] += 1
                    log.warning("mic send_realtime_input failed: %s", e)
                    raise
        finally:
            diag_task.cancel()
            try:
                await diag_task
            except (asyncio.CancelledError, Exception):
                pass
 
    # ---- camera -> Gemini -----------------------------------------------
    async def _video_task(self, session) -> None:
        if self._frame_source is None:
            return
        while not self._stop.is_set():
            try:
                jpeg = self._frame_source()
            except Exception as e:
                log.warning("frame_source raised: %s", e)
                jpeg = None
            if jpeg:
                try:
                    await session.send_realtime_input(
                        video=types.Blob(
                            data=jpeg,
                            mime_type="image/jpeg",
                        ),
                    )
                except Exception as e:
                    log.warning("video send failed: %s", e)
            await asyncio.sleep(self._video_period)
 
    # ---- Gemini -> speaker ---------------------------------------------
    async def _receive_task(self, session) -> None:
        """Read server messages, resample 24->16 kHz, feed ``speaker_sink``.
 
        A small asyncio.Queue decouples the websocket reader from the
        speaker sink so network backpressure never blocks the read loop.
        """
        playback_q: asyncio.Queue = asyncio.Queue(maxsize=256)
        _SENTINEL = b""
 
        stats = {
            "recv_chunks": 0, "recv_bytes": 0,
            "played_chunks": 0, "played_samples": 0,
            "dropped": 0, "queue_peak": 0,
            "turns": 0, "interruptions": 0,
            "sink_fail": 0,
        }
 
        def _to_mono_f32_16k(chunk: bytes) -> np.ndarray:
            """Gemini int16 @ 24 kHz mono -> float32 @ 16 kHz mono."""
            arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            arr /= INT16_SCALE
            # Polyphase resample 24 kHz -> 16 kHz, ratio 2/3.
            return resample_poly(arr, 2, 3).astype(np.float32)
 
        sink = self._speaker_sink
 
        async def playback_worker() -> None:
            while True:
                chunk = await playback_q.get()
                if chunk is _SENTINEL:
                    return
                try:
                    samples = await asyncio.to_thread(_to_mono_f32_16k, chunk)
                    if sink is not None:
                        await asyncio.to_thread(sink, samples)
                    stats["played_chunks"] += 1
                    stats["played_samples"] += int(samples.shape[0])
                except Exception as e:
                    stats["sink_fail"] += 1
                    log.debug("speaker_sink failed: %s", e)
 
        worker = asyncio.create_task(playback_worker(), name="gemini-playback")
 
        async def diag_loop():
            last = time.monotonic()
            while not self._stop.is_set():
                try:
                    await asyncio.sleep(DIAG_INTERVAL)
                except asyncio.CancelledError:
                    return
                now = time.monotonic()
                dt = max(now - last, 1e-3)
                last = now
                rc = stats["recv_chunks"]; stats["recv_chunks"] = 0
                rb = stats["recv_bytes"];  stats["recv_bytes"] = 0
                pc = stats["played_chunks"]; stats["played_chunks"] = 0
                ps = stats["played_samples"]; stats["played_samples"] = 0
                qpeak = stats["queue_peak"]; stats["queue_peak"] = 0
                dropped = stats["dropped"]; stats["dropped"] = 0
                turns = stats["turns"]; stats["turns"] = 0
                ints = stats["interruptions"]; stats["interruptions"] = 0
                sink_fail = stats["sink_fail"]; stats["sink_fail"] = 0
                recv_sec = rb / (GEMINI_OUTPUT_SR * 2.0)
                played_sec = ps / SINK_SR
                log.info(
                    "SPK  recv=%d (%.2fs audio) played=%d (%.2fs) q_peak=%d "
                    "dropped=%d sink_fail=%d turns=%d barge_in=%d",
                    rc, recv_sec, pc, played_sec, qpeak, dropped,
                    sink_fail, turns, ints,
                )
 
        diag_task = asyncio.create_task(diag_loop(), name="spk-diag")
        try:
            while not self._stop.is_set():
                async for msg in session.receive():
                    if self._stop.is_set():
                        break
 
                    # Session-resumption handle (needed for the 10-min cap).
                    sru = msg.session_resumption_update
                    if sru and sru.resumable and sru.new_handle:
                        self._resumption_handle = sru.new_handle
 
                    if msg.go_away is not None:
                        log.info(
                            "Gemini Live go_away received (time_left=%s); will reconnect",
                            getattr(msg.go_away, "time_left", None),
                        )
 
                    # ``msg.data`` concatenates all inline_data parts.
                    data = msg.data
                    if data:
                        self.is_speaking.set()
                        # Hold the mic gate while we're streaming audio out,
                        # plus a tail so the speaker buffer can drain and
                        # room echo can decay before we listen again.
                        self._mic_gate_until = time.monotonic() + self._mic_gate_tail_s
                        stats["recv_chunks"] += 1
                        stats["recv_bytes"] += len(data)
                        try:
                            playback_q.put_nowait(data)
                        except asyncio.QueueFull:
                            stats["dropped"] += 1
                            try:
                                playback_q.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            playback_q.put_nowait(data)
                        qsize = playback_q.qsize()
                        if qsize > stats["queue_peak"]:
                            stats["queue_peak"] = qsize
 
                    # Tool calls arrive as a top-level field on the message,
                    # independently of ``server_content``. Dispatch before
                    # the server_content early-exit so we never drop them.
                    tc = getattr(msg, "tool_call", None)
                    if tc is not None and getattr(tc, "function_calls", None):
                        # Dispatch tool calls in a background task so the
                        # receive loop keeps pulling audio chunks while
                        # long-running handlers (e.g. face enrollment)
                        # are still working.
                        task = asyncio.create_task(
                            self._dispatch_tool_call(session, tc),
                            name="gemini-tool-dispatch",
                        )
                        self._tool_tasks.add(task)
                        task.add_done_callback(self._tool_tasks.discard)

                    sc = msg.server_content
                    if sc is None:
                        continue
                    if sc.interrupted:
                        # User barged in: drop anything not yet handed to the
                        # sink so the old reply doesn't keep playing over them.
                        self.is_speaking.clear()
                        stats["interruptions"] += 1
                        drained = 0
                        while not playback_q.empty():
                            try:
                                playback_q.get_nowait()
                                drained += 1
                            except asyncio.QueueEmpty:
                                break
                        log.info(
                            "Gemini interrupted (barge-in); drained %d chunks",
                            drained,
                        )
                    if sc.turn_complete:
                        self.is_speaking.clear()
                        stats["turns"] += 1
                # ``async for`` ended without an exception: the server
                # closed this turn's stream. Re-enter for the next turn.
                log.debug(
                    "session.receive() generator exhausted; reopening for next turn"
                )
        finally:
            diag_task.cancel()
            try:
                await diag_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                playback_q.put_nowait(_SENTINEL)
            except asyncio.QueueFull:
                worker.cancel()
            try:
                await asyncio.wait_for(worker, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                worker.cancel()

    # ---- tool calls -----------------------------------------------------
    async def _dispatch_tool_call(self, session, tool_call) -> None:
        """Run each function call through ``self._tool_handler`` and reply.

        Gemini Live pauses the model turn until the client posts a
        ``tool_response`` for every pending call, so we make sure to
        always send a response -- even if the handler crashes or no
        handler is configured.
        """
        responses = []
        for fc in tool_call.function_calls:
            name = getattr(fc, "name", "") or ""
            # ``fc.args`` is a dict-like proto; coerce to a plain dict so
            # handlers can use the usual ``.get`` / subscript API.
            raw_args = getattr(fc, "args", None) or {}
            try:
                args: Dict[str, Any] = dict(raw_args)
            except Exception:
                args = {}

            t0 = time.monotonic()
            print(f"\n[TOOL CALL START] {name}({args})  t={t0:.6f}")

            if self._tool_handler is None:
                log.warning("tool_call '%s' but no tool_handler registered", name)
                result: Dict[str, Any] = {
                    "ok": False,
                    "error": "no tool handler configured",
                }
            else:
                try:
                    result = await self._tool_handler(name, args)
                except Exception as e:
                    log.exception("tool_handler('%s') crashed: %s", name, e)
                    result = {"ok": False, "error": str(e)}

            if not isinstance(result, dict):
                result = {"ok": True, "value": result}

            elapsed = time.monotonic() - t0
            print(f"[TOOL CALL DONE ] {name} -> {result}  elapsed={elapsed:.3f}s\n")
            log.info("tool_call %s(%s) -> %s  (%.3fs)", name, args, result, elapsed)

            responses.append(
                types.FunctionResponse(
                    id=getattr(fc, "id", None),
                    name=name,
                    response=result,
                )
            )

        try:
            await session.send_tool_response(function_responses=responses)
        except Exception as e:
            log.warning("send_tool_response failed: %s", e)