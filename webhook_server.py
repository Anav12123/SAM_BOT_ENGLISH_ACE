# """
# webhook_server.py
# - Interrupt path: trigger and LLM run sequentially (not parallel) so we can
#   pass was_interrupted=True and the correct context to the LLM
# - Normal path: still parallel for speed
# """

# import asyncio
# import json
# import time
# import base64
# from aiohttp import web
# from collections import deque

# from Trigger import TriggerDetector
# from Agent import PMAgent
# from Speaker import CartesiaSpeaker, _mix_noise, get_duration_ms


# def ts():
#     return time.strftime("%H:%M:%S")

# def elapsed(since: float) -> str:
#     return f"{(time.time() - since)*1000:.0f}ms"


# BOT_NAMES = {"sam", "bot", "recall bot"}


# class WebhookServer:
#     def __init__(self, port: int = 8000, bot_id: str = None):
#         self.port    = port
#         self.trigger = TriggerDetector()
#         self.agent   = PMAgent()
#         self.speaker = CartesiaSpeaker(bot_id=bot_id)

#         self._processing            = False
#         self._speaking_until: float = 0.0
#         self._current_task: asyncio.Task | None = None

#         # What Sam was mid-saying when interrupted
#         self._interrupted_response: str = ""

#         self._buffer      = []
#         self._buffer_task = None

#         self._pending: list[tuple[str, str, float]] = []
#         self._pending_task: asyncio.Task | None = None

#         self._convo_history: deque[str] = deque(maxlen=6)

#         self.app = web.Application()
#         self.app.router.add_post("/webhook", self.handle_webhook)
#         self.app.router.add_get("/health",   self.handle_health)

#     def _sam_is_speaking(self) -> bool:
#         return time.time() < self._speaking_until

#     def _interrupt(self, current_response: str = ""):
#         """Stop Sam. Save what he was saying for interrupt context."""
#         if self._sam_is_speaking():
#             print(f"[{ts()}] 🛑 Interrupted mid-speech")
#         if current_response:
#             self._interrupted_response = current_response
#         self._speaking_until = 0.0
#         if self._current_task and not self._current_task.done():
#             self._current_task.cancel()
#             self._current_task = None

#     def _build_interrupt_context(self, new_question: str) -> str:
#         """
#         Builds the full context string passed to the LLM when Sam is interrupted.
#         Includes last 3 exchanges + what Sam was saying when cut off.
#         """
#         lines = list(self._convo_history)

#         # Last 3 exchanges = up to 6 lines (user + Sam pairs)
#         recent = lines[-6:] if len(lines) >= 6 else lines
#         context_block = "\n".join(recent) if recent else "No prior context."

#         interrupted_note = ""
#         if self._interrupted_response:
#             interrupted_note = (
#                 f"\nYou were mid-sentence saying: \"{self._interrupted_response}\" "
#                 f"when the user interrupted. Acknowledge this naturally with one short "
#                 f"pivot phrase, then answer their new question."
#             )

#         return (
#             f"Recent conversation (last 3 exchanges):\n{context_block}"
#             f"{interrupted_note}\n\n"
#             f"User just interrupted and asked: \"{new_question}\""
#         )

#     async def handle_health(self, request: web.Request) -> web.Response:
#         return web.json_response({"status": "ok"})

#     async def handle_webhook(self, request: web.Request) -> web.Response:
#         t = time.time()
#         body = await request.text()
#         try:
#             payload = json.loads(body)
#         except Exception:
#             return web.json_response({"error": "invalid json"}, status=400)

#         if payload.get("event") != "transcript.data":
#             return web.json_response({"status": "ignored"})

#         inner   = payload.get("data", {}).get("data", {})
#         words   = inner.get("words", [])
#         text    = " ".join(w.get("text", "") for w in words).strip()
#         speaker = inner.get("participant", {}).get("name", "Unknown")

#         if not text:
#             return web.json_response({"status": "empty"})

#         if speaker.lower() in BOT_NAMES:
#             return web.json_response({"status": "self"})

#         print(f"\n[{ts()}] [{speaker}] {text}  ⏱ {elapsed(t)}")

#         if self._sam_is_speaking():
#             self._interrupt()  # stop Sam, _interrupted_response already set in _process
#             self._pending.append((speaker, text, t))
#             print(f"[{ts()}] Queued interrupt: \"{text}\" (pending={len(self._pending)})")
#             if self._pending_task and not self._pending_task.done():
#                 self._pending_task.cancel()
#             self._pending_task = asyncio.create_task(self._flush_pending_after_lock())
#             return web.json_response({"status": "interrupted"})

#         self._buffer.append((speaker, text, t))
#         if self._buffer_task and not self._buffer_task.done():
#             self._buffer_task.cancel()
#         self._buffer_task = asyncio.create_task(
#             self._flush_after_silence(speaker, t)
#         )
#         return web.json_response({"status": "ok"})

#     async def _flush_after_silence(self, speaker: str, t0: float):
#         try:
#             await asyncio.sleep(1.2)
#         except asyncio.CancelledError:
#             return

#         if not self._buffer:
#             return

#         full_text = " ".join(txt for _, txt, _ in self._buffer)
#         self._buffer.clear()
#         self._convo_history.append(f"{speaker}: {full_text}")
#         print(f"[{ts()}] Buffered: \"{full_text}\"")

#         task = asyncio.create_task(
#             self._process(full_text, speaker, t0, was_interrupted=False)
#         )
#         self._current_task = task

#     async def _flush_pending_after_lock(self):
#         try:
#             while self._sam_is_speaking():
#                 await asyncio.sleep(0.05)
#             await asyncio.sleep(1.2)
#         except asyncio.CancelledError:
#             return

#         if not self._pending:
#             return

#         speaker  = self._pending[-1][0]
#         t0       = self._pending[0][2]
#         combined = " ".join(txt for _, txt, _ in self._pending)
#         self._pending.clear()

#         print(f"[{ts()}] Flushing interrupt: \"{combined}\"")
#         self._convo_history.append(f"{speaker}: {combined}")

#         task = asyncio.create_task(
#             self._process(combined, speaker, t0, was_interrupted=True)
#         )
#         self._current_task = task

#     async def _process(self, text: str, speaker: str, t0: float, was_interrupted: bool = False):
#         if self._processing or self._sam_is_speaking():
#             print(f"[{ts()}] Dropping — Sam is still speaking")
#             return

#         self._processing = True
#         try:
#             t1 = time.time()

#             if was_interrupted:
#                 # ── INTERRUPT PATH ──────────────────────────────────────────
#                 # Build rich context FIRST (last 3 exchanges + what Sam was saying)
#                 # then run trigger + LLM sequentially so we pass the right args
#                 print(f"[{ts()}] Mode: INTERRUPTED")
#                 interrupt_context = self._build_interrupt_context(text)

#                 # Trigger still runs — but for direct interrupts it almost always YES
#                 memory_snapshot = self.agent.memory[-30:]
#                 should = await self.trigger.should_respond(
#                     text, speaker, interrupt_context, memory_snapshot
#                 )
#                 print(f"[{ts()}] Trigger: {'YES' if should else 'NO'} ({elapsed(t1)})")

#                 if not should:
#                     return

#                 # LLM gets interrupt_context + interrupted=True flag
#                 response = await self.agent.respond_with_context(
#                     user_text=text,
#                     context=interrupt_context,
#                     interrupted=True,
#                 )

#             else:
#                 # ── NORMAL PATH ─────────────────────────────────────────────
#                 # Trigger + LLM in parallel for speed
#                 print(f"[{ts()}] Trigger + LLM in parallel...")
#                 normal_context  = "\n".join(self._convo_history)
#                 memory_snapshot = self.agent.memory[-30:]

#                 trigger_task = asyncio.create_task(
#                     self.trigger.should_respond(
#                         text, speaker, normal_context, memory_snapshot
#                     )
#                 )
#                 llm_task = asyncio.create_task(
#                     self.agent.respond_with_context(
#                         user_text=text,
#                         context=normal_context,
#                         interrupted=False,
#                     )
#                 )

#                 should = await trigger_task
#                 print(f"[{ts()}] Trigger: {'YES' if should else 'NO'} ({elapsed(t1)})")

#                 if not should:
#                     llm_task.cancel()
#                     return

#                 response = await llm_task

#             print(f"[{ts()}] LLM {elapsed(t1)}: \"{response}\"")
#             self._convo_history.append(f"Sam: {response}")
#             self._interrupted_response = ""  # clear after a full response
#             self.trigger.mark_responded()

#             # ── TTS ─────────────────────────────────────────────────────────
#             t2 = time.time()
#             print(f"[{ts()}] TTS...")

#             # Save response as current so interrupt can capture it
#             self._interrupted_response = response

#             voice_bytes = await self.speaker._synthesise(response)
#             print(f"[{ts()}] TTS done {elapsed(t2)}")

#             loop       = asyncio.get_event_loop()
#             word_count = len(response.split())
#             if self.speaker._noise_slices and word_count > 6:
#                 t3 = time.time()
#                 audio_bytes, duration_ms = await loop.run_in_executor(
#                     None, _mix_noise, voice_bytes, self.speaker._noise_slices, response
#                 )
#                 print(f"[{ts()}] Noise {elapsed(t3)}")
#             else:
#                 audio_bytes = voice_bytes
#                 duration_ms = await loop.run_in_executor(
#                     None, get_duration_ms, audio_bytes
#                 )

#             t4 = time.time()
#             b64 = await loop.run_in_executor(
#                 None, lambda: base64.b64encode(audio_bytes).decode("utf-8")
#             )

#             duration_s = duration_ms / 1000.0
#             self._speaking_until = time.time() + duration_s + 0.8
#             print(f"[{ts()}] Speaking lock: {duration_s:.1f}s + 0.8s tail")

#             await self.speaker._inject_into_meeting(b64)
#             print(f"[{ts()}] Inject {elapsed(t4)} | TOTAL {elapsed(t0)}")

#             remaining = self._speaking_until - time.time()
#             if remaining > 0:
#                 await asyncio.sleep(remaining)

#             # Audio finished — clear the interrupted response
#             self._interrupted_response = ""

#         except asyncio.CancelledError:
#             # _interrupted_response intentionally kept — contains what Sam was saying
#             print(f"[{ts()}] _process cancelled — saving partial response for interrupt context")
#             self._speaking_until = 0.0

#         finally:
#             self._processing = False

#     async def start(self):
#         runner = web.AppRunner(self.app)
#         await runner.setup()
#         site = web.TCPSite(runner, "0.0.0.0", self.port)
#         await site.start()
#         print(f"[{ts()}] Server ready on :8000\n")

"""
webhook_server.py
Key fix: pre-process pending while Sam is still speaking.
When user speaks mid-Sam:
  1. Collect their words into _pending (debounced 0.6s silence)
  2. Immediately start LLM + TTS in background
  3. Hold the result until Sam's audio finishes
  4. Inject instantly the moment Sam stops — zero gap
"""

import asyncio
import json
import time
import base64
from aiohttp import web
from collections import deque

from Trigger import TriggerDetector
from Agent import PMAgent
from Speaker import CartesiaSpeaker, get_duration_ms


def ts():
    return time.strftime("%H:%M:%S")

def elapsed(since: float) -> str:
    return f"{(time.time() - since)*1000:.0f}ms"


BOT_NAMES = {"sam", "bot", "recall bot"}


class WebhookServer:
    def __init__(self, port: int = 8000, bot_id: str = None):
        self.port    = port
        self.trigger = TriggerDetector()
        self.agent   = PMAgent()
        self.speaker = CartesiaSpeaker(bot_id=bot_id)

        self._processing            = False
        self._speaking_until: float = 0.0
        self._current_task: asyncio.Task | None = None
        self._last_response: str    = ""

        self._buffer:      list = []
        self._buffer_task: asyncio.Task | None = None

        # Pending: words spoken while Sam is playing
        self._pending:      list[tuple[str, str, float]] = []
        self._pending_debounce: asyncio.Task | None = None

        # Pre-processed audio ready to fire the moment Sam finishes
        self._prepped_audio: bytes | None = None   # raw voice bytes
        self._prepped_b64:   str   | None = None   # base64 encoded
        self._prepped_dur_ms: int  = 0             # duration ms
        self._prepped_for:   str   = ""            # the text it responds to
        self._prep_task:     asyncio.Task | None = None

        self._convo_history: deque[str] = deque(maxlen=6)

        self.app = web.Application()
        self.app.router.add_post("/webhook", self.handle_webhook)
        self.app.router.add_get("/health",   self.handle_health)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sam_is_speaking(self) -> bool:
        return time.time() < self._speaking_until

    def _memory_strings(self) -> list[str]:
        result = []
        for item in self.agent.memory[-10:]:
            result.append(item[0] if isinstance(item, tuple) else str(item))
        return result

    def _build_interrupt_context(self, pending_text: str) -> str:
        lines  = list(self._convo_history)
        recent = lines[-6:] if len(lines) >= 6 else lines
        context_block = "\n".join(recent) if recent else "No prior context."
        interrupted_note = (
            f"\nYou were saying: \"{self._last_response}\" when interrupted."
        ) if self._last_response else ""
        return (
            f"Last 3 exchanges:\n{context_block}"
            f"{interrupted_note}\n\n"
            f"User said while you spoke: \"{pending_text}\""
        )

    # ── Webhook ───────────────────────────────────────────────────────────────

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_webhook(self, request: web.Request) -> web.Response:
        t = time.time()
        body = await request.text()
        try:
            payload = json.loads(body)
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        if payload.get("event") != "transcript.data":
            return web.json_response({"status": "ignored"})

        inner   = payload.get("data", {}).get("data", {})
        words   = inner.get("words", [])
        text    = " ".join(w.get("text", "") for w in words).strip()
        speaker = inner.get("participant", {}).get("name", "Unknown")

        if not text:
            return web.json_response({"status": "empty"})

        if speaker.lower() in BOT_NAMES:
            return web.json_response({"status": "self"})

        print(f"\n[{ts()}] [{speaker}] {text}  ⏱ {elapsed(t)}")

        if self._sam_is_speaking():
            # Collect pending words
            self._pending.append((speaker, text, t))
            print(f"[{ts()}] Queued (Sam speaking): \"{text}\" (pending={len(self._pending)})")

            # Debounce — restart silence timer on each new word
            if self._pending_debounce and not self._pending_debounce.done():
                self._pending_debounce.cancel()
            self._pending_debounce = asyncio.create_task(
                self._debounce_then_prep(speaker, t)
            )
            return web.json_response({"status": "queued"})

        # Normal path
        self._buffer.append((speaker, text, t))
        if self._buffer_task and not self._buffer_task.done():
            self._buffer_task.cancel()
        self._buffer_task = asyncio.create_task(
            self._flush_after_silence(speaker, t)
        )
        return web.json_response({"status": "ok"})

    # ── Pending debounce → pre-process ────────────────────────────────────────

    async def _debounce_then_prep(self, speaker: str, t0: float):
        """
        Wait for user to finish speaking (0.6s silence),
        then immediately kick off LLM+TTS in background while Sam is still playing.
        When Sam finishes, _fire_prepped() injects the ready audio with zero delay.
        """
        try:
            await asyncio.sleep(0.6)
        except asyncio.CancelledError:
            return  # more words came in, timer restarted

        if not self._pending:
            return

        speaker  = self._pending[-1][0]
        t0       = self._pending[0][2]
        combined = " ".join(txt for _, txt, _ in self._pending)
        self._pending.clear()

        print(f"[{ts()}] User finished: \"{combined}\" — pre-processing now")
        self._convo_history.append(f"{speaker}: {combined}")

        # Cancel any previous prep task
        if self._prep_task and not self._prep_task.done():
            self._prep_task.cancel()

        # Clear previous prepped audio
        self._prepped_b64    = None
        self._prepped_dur_ms = 0
        self._prepped_for    = combined

        # Start LLM+TTS immediately, hold result until Sam finishes
        self._prep_task = asyncio.create_task(
            self._preprocess_response(combined, speaker, t0)
        )

    async def _preprocess_response(self, text: str, speaker: str, t0: float):
        """
        Run LLM + TTS while Sam is still speaking.
        Store result in _prepped_b64 / _prepped_dur_ms.
        Then wait for Sam to finish and inject immediately.
        """
        try:
            t1 = time.time()
            context         = self._build_interrupt_context(text)
            memory_snapshot = self._memory_strings()

            # Check trigger
            should = await self.trigger.should_respond(
                text, speaker, context, memory_snapshot
            )
            print(f"[{ts()}] Pre-trigger: {'YES' if should else 'NO'} ({elapsed(t1)})")
            if not should:
                return

            # LLM
            response = await self.agent.respond_with_context(
                user_text=text,
                context=context,
                interrupted=True,
            )
            print(f"[{ts()}] Pre-LLM {elapsed(t1)}: \"{response}\"")

            # TTS
            t2 = time.time()
            voice_bytes = await self.speaker._synthesise(response)
            print(f"[{ts()}] Pre-TTS {elapsed(t2)}")

            loop        = asyncio.get_event_loop()
            duration_ms = await loop.run_in_executor(None, get_duration_ms, voice_bytes)
            b64         = await loop.run_in_executor(
                None, lambda: base64.b64encode(voice_bytes).decode("utf-8")
            )

            # Store ready audio
            self._prepped_b64    = b64
            self._prepped_dur_ms = duration_ms
            self._prepped_for    = response

            # If Sam already finished while we were processing, fire immediately
            if not self._sam_is_speaking() and not self._processing:
                print(f"[{ts()}] Sam already done — firing prepped response immediately")
                await self._fire_prepped(response, speaker, t0)
            else:
                # Sam still speaking — schedule fire for when he finishes
                print(f"[{ts()}] Prepped and ready — waiting for Sam to finish")
                asyncio.create_task(self._wait_and_fire(response, speaker, t0))

        except asyncio.CancelledError:
            print(f"[{ts()}] Pre-process cancelled")
        except Exception as e:
            print(f"[{ts()}] Pre-process error: {e}")

    async def _wait_and_fire(self, response: str, speaker: str, t0: float):
        """Poll until Sam finishes, then fire prepped audio instantly."""
        while self._sam_is_speaking():
            await asyncio.sleep(0.03)  # poll every 30ms

        if self._prepped_b64 is None:
            return  # was cleared (e.g. newer prep replaced it)

        await self._fire_prepped(response, speaker, t0)

    async def _fire_prepped(self, response: str, speaker: str, t0: float):
        """Inject pre-built audio immediately. No LLM/TTS wait."""
        if self._prepped_b64 is None:
            return
        if self._processing or self._sam_is_speaking():
            return

        self._processing = True
        try:
            b64         = self._prepped_b64
            duration_ms = self._prepped_dur_ms

            # Clear so it can't fire twice
            self._prepped_b64    = None
            self._prepped_dur_ms = 0

            self._convo_history.append(f"Sam: {response}")
            self._last_response = response
            self.trigger.mark_responded()

            duration_s           = duration_ms / 1000.0
            self._speaking_until = time.time() + duration_s + 0.3
            print(f"[{ts()}] 🚀 Firing prepped: {duration_s:.1f}s | TOTAL {elapsed(t0)}")

            await self.speaker._inject_into_meeting(b64)

            remaining = self._speaking_until - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)

        except Exception as e:
            print(f"[{ts()}] Fire error: {e}")
            self._speaking_until = 0.0
        finally:
            self._processing    = False
            self._speaking_until = 0.0
            self._last_response  = ""   # clear so it never bleeds into next interrupt

    # ── Normal path ───────────────────────────────────────────────────────────

    async def _flush_after_silence(self, speaker: str, t0: float):
        try:
            await asyncio.sleep(0.6)
        except asyncio.CancelledError:
            return
        if not self._buffer:
            return
        full_text = " ".join(txt for _, txt, _ in self._buffer)
        self._buffer.clear()
        self._convo_history.append(f"{speaker}: {full_text}")
        print(f"[{ts()}] Buffered: \"{full_text}\"")
        task = asyncio.create_task(
            self._process(full_text, speaker, t0, was_interrupted=False)
        )
        self._current_task = task

    async def _process(
        self, text: str, speaker: str, t0: float, was_interrupted: bool = False
    ):
        if self._processing or self._sam_is_speaking():
            print(f"[{ts()}] Dropping — Sam is still speaking")
            return

        self._processing = True
        try:
            t1 = time.time()
            print(f"[{ts()}] Trigger + LLM in parallel...")
            context         = "\n".join(self._convo_history)
            memory_snapshot = self._memory_strings()

            trigger_task = asyncio.create_task(
                self.trigger.should_respond(text, speaker, context, memory_snapshot)
            )
            llm_task = asyncio.create_task(
                self.agent.respond_with_context(
                    user_text=text, context=context, interrupted=False,
                )
            )

            should = await trigger_task
            print(f"[{ts()}] Trigger: {'YES' if should else 'NO'} ({elapsed(t1)})")
            if not should:
                llm_task.cancel()
                return

            response = await llm_task
            print(f"[{ts()}] LLM {elapsed(t1)}: \"{response}\"")
            self._convo_history.append(f"Sam: {response}")
            self._last_response = response
            self.trigger.mark_responded()

            t2 = time.time()
            print(f"[{ts()}] TTS...")
            voice_bytes = await self.speaker._synthesise(response)
            print(f"[{ts()}] TTS {elapsed(t2)}")

            loop        = asyncio.get_event_loop()
            duration_ms = await loop.run_in_executor(None, get_duration_ms, voice_bytes)
            b64         = await loop.run_in_executor(
                None, lambda: base64.b64encode(voice_bytes).decode("utf-8")
            )

            duration_s           = duration_ms / 1000.0
            self._speaking_until = time.time() + duration_s + 0.3
            print(f"[{ts()}] Lock: {duration_s:.1f}s | TOTAL {elapsed(t0)}")

            await self.speaker._inject_into_meeting(b64)

            remaining = self._speaking_until - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)

        except asyncio.CancelledError:
            print(f"[{ts()}] _process cancelled")
        except Exception as e:
            print(f"[{ts()}] _process error: {e}")
            self._speaking_until = 0.0
        finally:
            self._processing    = False
            self._speaking_until = 0.0
            self._last_response  = ""   # clear after each full response

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        print(f"[{ts()}] Server ready on :8000\n")