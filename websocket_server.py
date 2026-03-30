"""
websocket_server.py — Streaming LLM + parallel TTS + filler words

Pipeline:
  Deepgram transcript → Trigger check → LLM streaming → parallel TTS → concat → inject
  If [SEARCH]: filler plays instantly → LLM converts query → SerpAPI → LLM streams summary
"""

import asyncio
import json
import time
import base64
import re as _re
from aiohttp import web
import aiohttp
from collections import deque

from Trigger import TriggerDetector
from Agent import PMAgent
from Speaker import CartesiaSpeaker, _mix_noise


def ts():
    return time.strftime("%H:%M:%S")

def elapsed(since: float) -> str:
    return f"{(time.time() - since)*1000:.0f}ms"

WORDS_PER_SECOND = 3.2

# ── Fix Deepgram misrecognitions ──────────────────────────────────────────
_TRANSCRIPTION_FIXES = [
    (_re.compile(r'\b(?:NF\s*Cloud|Enuf\s*Cloud|Enough\s*Cloud|Nav\s*Cloud|Anav\s*Cloud|Arnav\s*Cloud|Anab\s*Cloud|NFClouds?|EnoughClouds?|NavClouds?|AnavCloud)\b', _re.IGNORECASE), 'AnavClouds'),
    (_re.compile(r'\b(?:Sales\s*Force|Sells\s*Force|Cells\s*Force|SalesForce)\b', _re.IGNORECASE), 'Salesforce'),
]

def _fix_transcription(text: str) -> str:
    result = text
    for pattern, replacement in _TRANSCRIPTION_FIXES:
        result = pattern.sub(replacement, result)
    if result != text:
        print(f"[Transcript Fix] \"{text}\" → \"{result}\"")
    return result


class WebSocketServer:
    def __init__(self, port: int = 8000, bot_id: str = None):
        self.port           = port
        self.trigger        = TriggerDetector()
        self.agent          = PMAgent()
        self.speaker        = CartesiaSpeaker(bot_id=bot_id)
        self._speaking      = False
        self._audio_playing = False
        self._convo_history = deque(maxlen=10)

        self._current_task:    asyncio.Task | None = None
        self._current_text:    str  = ""
        self._current_speaker: str  = ""
        self._interrupt_event: asyncio.Event = asyncio.Event()
        self._generation:      int  = 0

        self._buffer:      list = []
        self._buffer_task: asyncio.Task | None = None

        # TTS rate limiter
        self._tts_semaphore = asyncio.Semaphore(2)

        self.app = web.Application()
        self.app.router.add_get("/ws",     self.handle_websocket)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "speaking": self._speaking})

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        print(f"[{ts()}] ✅ Recall.ai WebSocket connected")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_event(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[{ts()}] ⚠️  WS error: {ws.exception()}")
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
        except Exception as e:
            print(f"[{ts()}] WS handler error: {e}")
        finally:
            print(f"[{ts()}] WebSocket disconnected")
        return ws

    async def _handle_event(self, raw: str):
        t = time.time()
        try:
            payload = json.loads(raw)
        except Exception:
            return

        event = payload.get("event", "")

        # ── Transcript ────────────────────────────────────────────────────
        if event == "transcript.data":
            inner   = payload.get("data", {}).get("data", {})
            words   = inner.get("words", [])
            text    = " ".join(w.get("text", "") for w in words).strip()
            speaker = inner.get("participant", {}).get("name", "Unknown")
            if not text or speaker.lower() == "sam":
                return

            text = _fix_transcription(text)
            print(f"\n[{ts()}] [{speaker}] {text}  ⏱ {elapsed(t)}")

            # Store in full meeting log for long-term memory
            self.agent._store_to_log(speaker, text)

            if self._buffer_task and not self._buffer_task.done():
                self._buffer_task.cancel()

            # Same speaker adds more while being processed
            if self._speaking and self._current_speaker == speaker:
                combined = f"{self._current_text} {text}".strip()
                print(f"[{ts()}] 🔄 Combined: \"{combined}\" — restarting")
                if self._current_task and not self._current_task.done():
                    self._current_task.cancel()
                if self._audio_playing:
                    asyncio.create_task(self.speaker.stop_audio())
                self._speaking = False
                self._audio_playing = False
                self._interrupt_event.set()
                await asyncio.sleep(0)
                self._start_process(combined, speaker, t)

            # Different speaker interrupts
            elif self._speaking and self._current_speaker != speaker:
                print(f"[{ts()}] ⚡ INTERRUPT — {speaker} cut in")
                asyncio.create_task(self.speaker.stop_audio())
                self._interrupt_event.set()
                self._start_process(text, speaker, t)

            # Free — start immediately
            else:
                self._start_process(text, speaker, t)

        # Speech events
        elif event == "participant_events.speech_off":
            speaker = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            print(f"[{ts()}] 🔇 {speaker} stopped speaking")
            if self._buffer and not self._speaking:
                full_text = " ".join(txt for _, txt, _ in self._buffer)
                t0 = self._buffer[0][2]
                self._buffer.clear()
                self._start_process(full_text, speaker, t0)
            self._buffer.clear()

        elif event == "participant_events.speech_on":
            speaker = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            print(f"[{ts()}] 🎤 {speaker} started speaking")
            if self._speaking and self._current_speaker != speaker:
                print(f"[{ts()}] ⚡ INTERRUPT (speech_on) — {speaker} cut in")
                asyncio.create_task(self.speaker.stop_audio())
                self._interrupt_event.set()

        elif event == "participant_events.join":
            name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            if name and name.lower() != "sam":
                print(f"[{ts()}] 👋 {name} joined")
                asyncio.create_task(self._greet(name, t))

        elif event == "participant_events.leave":
            name = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            if name and name.lower() != "sam":
                print(f"[{ts()}] 👋 {name} left")

    def _start_process(self, text, speaker, t0):
        self._generation     += 1
        my_gen                = self._generation
        self._current_text    = text
        self._current_speaker = speaker
        self._interrupt_event.clear()
        self._current_task = asyncio.create_task(self._process(text, speaker, t0, my_gen))

    async def _greet(self, name, t0):
        await asyncio.sleep(2.0)
        if self._speaking:
            return
        greeting = f"Hey {name}, welcome to the call!"
        self._log_sam(greeting)
        await self._speak_response(greeting, t0)

    def _log_sam(self, text: str):
        """Store Sam's response in both convo history and meeting log."""
        self._convo_history.append(f"Sam: {text}")
        self.agent._store_to_log("Sam", text)

    async def _tts(self, text: str) -> bytes:
        async with self._tts_semaphore:
            return await self.speaker._synthesise(text)

    # ══════════════════════════════════════════════════════════════════════════
    # Main processing pipeline
    # ══════════════════════════════════════════════════════════════════════════

    async def _process(self, text, speaker, t0, generation=0):
        if self._speaking:
            print(f"[{ts()}] ⚠️  Already speaking — dropping")
            return

        self._speaking = True
        self._interrupt_event.clear()
        my_gen = generation

        try:
            context = "\n".join(self._convo_history)
            memory  = [m[0] for m in self.agent.memory[-20:]]
            t1 = time.time()

            # Fire trigger + LLM in parallel
            sentence_queue = asyncio.Queue()
            llm_task = asyncio.create_task(
                self.agent.stream_sentences_to_queue(text, context, sentence_queue)
            )
            trigger_task = asyncio.create_task(
                self.trigger.should_respond(text, speaker, context, memory)
            )

            print(f"[{ts()}] Trigger + LLM streaming in parallel...")
            should = await trigger_task
            print(f"[{ts()}] Trigger: {'YES' if should else 'NO'} ({elapsed(t1)})")

            if not should:
                llm_task.cancel()
                return

            # Collect sentences + TTS
            sentences:      list[str] = []
            tts_tasks:      list[asyncio.Task] = []
            all_sentences:  list[str] = []

            while True:
                if self._interrupt_event.is_set() or my_gen != self._generation:
                    print(f"[{ts()}] ⚡ Superseded — aborting")
                    llm_task.cancel()
                    for t_task in tts_tasks:
                        t_task.cancel()
                    return

                try:
                    item = await asyncio.wait_for(sentence_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    print(f"[{ts()}] ⚠️  LLM queue timeout")
                    break

                if item is None:
                    break

                # Filler flush — inject immediately
                if item == "__FLUSH__":
                    if sentences and tts_tasks:
                        print(f"[{ts()}] 🗣️ Flushing filler audio...")
                        flush_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
                        chunks = [r for r in flush_results if not isinstance(r, Exception)]
                        if chunks:
                            filler_audio = b"".join(chunks)
                            filler_words = sum(len(s.split()) for s in sentences)
                            filler_dur   = max(500, filler_words * 300)
                            filler_b64   = base64.b64encode(filler_audio).decode("utf-8")

                            if not (self._interrupt_event.is_set() or my_gen != self._generation):
                                tf = time.time()
                                await self.speaker._inject_into_meeting(filler_b64)
                                self._audio_playing = True
                                print(f"[{ts()}] 🗣️ Filler injected ({elapsed(tf)}) | TOTAL {elapsed(t0)}")

                                try:
                                    await asyncio.wait_for(self._interrupt_event.wait(), timeout=filler_dur/1000)
                                    print(f"[{ts()}] ⚡ Interrupted during filler")
                                    all_sentences.extend(sentences)
                                    self._log_sam(f"{' '.join(all_sentences)} [interrupted]")
                                    self.trigger.mark_responded()
                                    llm_task.cancel()
                                    return
                                except asyncio.TimeoutError:
                                    pass
                                self._audio_playing = False

                        all_sentences.extend(sentences)
                        sentences.clear()
                        tts_tasks.clear()
                    continue

                # Normal sentence
                sentences.append(item)
                idx = len(all_sentences) + len(sentences)
                print(f"[{ts()}] LLM sentence {idx} ({elapsed(t1)}): \"{item}\"")
                tts_tasks.append(asyncio.create_task(self._tts(item)))

            if not sentences and not tts_tasks:
                if all_sentences:
                    self._log_sam(' '.join(all_sentences))
                    self.trigger.mark_responded()
                return

            all_sentences.extend(sentences)

            # Await TTS
            t2 = time.time()
            print(f"[{ts()}] TTS ({len(sentences)} sentences, parallel+streamed)...")
            results = await asyncio.gather(*tts_tasks, return_exceptions=True)
            audio_chunks = [r for r in results if not isinstance(r, Exception)]

            if not audio_chunks:
                print(f"[{ts()}] ⚠️  All TTS failed")
                return

            if self._interrupt_event.is_set() or my_gen != self._generation:
                return

            tts_ms = (time.time() - t2) * 1000

            # Concat + inject
            audio_bytes  = b"".join(audio_chunks)
            full_resp    = " ".join(all_sentences)
            word_count   = sum(len(s.split()) for s in sentences)
            audio_dur_ms = max(500, word_count * 300)

            loop = asyncio.get_event_loop()
            b64 = await loop.run_in_executor(
                None, lambda ab=audio_bytes: base64.b64encode(ab).decode("utf-8")
            )

            if self._interrupt_event.is_set() or my_gen != self._generation:
                return

            t3 = time.time()
            await self.speaker._inject_into_meeting(b64)
            self._audio_playing = True
            inject_ms = (time.time() - t3) * 1000
            print(f"[{ts()}] TTS {tts_ms:.0f}ms | Inject {inject_ms:.0f}ms | Lock {audio_dur_ms/1000:.1f}s | TOTAL {elapsed(t0)}")

            # Interruptible wait
            already = (time.time() - t2) * 1000
            wait_ms = max(100, audio_dur_ms - already)
            try:
                await asyncio.wait_for(self._interrupt_event.wait(), timeout=wait_ms/1000)
                self._log_sam(f"{full_resp} [interrupted]")
                self.trigger.mark_responded()
                return
            except asyncio.TimeoutError:
                pass

            self._audio_playing = False
            self._log_sam(full_resp)
            self.trigger.mark_responded()
            print(f"[{ts()}] ✅ Done")

        except asyncio.CancelledError:
            print(f"[{ts()}] 🔄 Task cancelled (new text combined)")
            try: llm_task.cancel()
            except: pass
            try:
                for t_task in tts_tasks:
                    t_task.cancel()
            except: pass
        except Exception as e:
            import traceback
            print(f"[{ts()}] ❌ _process error: {e}")
            traceback.print_exc()
        finally:
            self._audio_playing = False
            self._speaking      = False

    async def _speak_response(self, text, t0):
        if self._speaking:
            return
        self._speaking = True
        try:
            voice_bytes = await self._tts(text)
            b64 = base64.b64encode(voice_bytes).decode("utf-8")
            await self.speaker._inject_into_meeting(b64)
            self._interrupt_event.clear()
            try:
                await asyncio.wait_for(
                    self._interrupt_event.wait(),
                    timeout=len(text.split()) / WORDS_PER_SECOND
                )
            except asyncio.TimeoutError:
                pass
        except Exception as e:
            print(f"[{ts()}] ⚠️  _speak_response error: {e}")
        finally:
            self._speaking = False

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        print(f"[{ts()}] WebSocket server ready on ws://0.0.0.0:{self.port}/ws")
        print(f"[{ts()}] Health check: http://localhost:{self.port}/health\n")
