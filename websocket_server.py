"""
websocket_server.py — Production-grade streaming voice pipeline

Pipeline:
  Deepgram → Buffer+debounce → Trigger+Router parallel →
    [PM] → LLM stream → sentence TTS → inject each ASAP
    [FT] → filler inject → background search+TTS → inject or save pending

Edge cases handled:
  - Ack detection during search ("Sure", "OK", "Go ahead") → ignored
  - Pending search queue → delivers after current response
  - No overlapping audio ever — sequential inject + playback wait
  - Same speaker continuation → rebuffer
  - Different speaker interrupt → stop audio + process new
  - Filler playback tracked as _speaking/_audio_playing
  - Graceful crash handling in finally blocks
"""

import asyncio
import json
import time
import base64
import re as _re
import random
from aiohttp import web
import aiohttp
from collections import deque

from Trigger import TriggerDetector
from Agent import PMAgent, FILLERS
from Speaker import CartesiaSpeaker, _mix_noise


def ts():
    return time.strftime("%H:%M:%S")

def elapsed(since: float) -> str:
    return f"{(time.time() - since)*1000:.0f}ms"

WORDS_PER_SECOND = 3.2

# ── Ack phrases — ignored during search ──────────────────────────────────────
_ACK_PHRASES = frozenset({
    "sure", "ok", "okay", "yeah", "yes", "go ahead", "alright",
    "right", "hmm", "mhm", "cool", "got it", "fine", "yep", "yup",
    "carry on", "go on", "continue", "waiting", "i'm waiting",
    "i am waiting", "no problem", "take your time", "np",
})

# ── Fix Deepgram misrecognitions ─────────────────────────────────────────────
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

def _is_ack(text: str) -> bool:
    """Check if text is purely acknowledgement phrases."""
    fragments = _re.split(r'[.!?,]+', text.strip().lower())
    return all(f.strip() in _ACK_PHRASES or f.strip() == "" for f in fragments) and text.strip() != ""


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

        # Search state
        self._searching = False
        # Pending: list of (query_text, prepare_task) where prepare_task returns list[(sentence, audio_bytes)]
        self._pending_searches: list[tuple[str, asyncio.Task]] = []

        # TTS rate limiter — matches number of Cartesia keys
        self._tts_semaphore = asyncio.Semaphore(4)

        self.app = web.Application()
        self.app.router.add_get("/ws",     self.handle_websocket)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_health(self, request):
        return web.json_response({"status": "ok", "speaking": self._speaking, "searching": self._searching})

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        print(f"[{ts()}] ✅ Recall.ai WebSocket connected")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await self._handle_event(msg.data)
                    except Exception as e:
                        print(f"[{ts()}] ⚠️  Event handler error: {e}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[{ts()}] ⚠️  WS error: {ws.exception()}")
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                    break
        except Exception as e:
            print(f"[{ts()}] WS handler error: {e}")
        finally:
            print(f"[{ts()}] WebSocket disconnected")
        return ws

    # ══════════════════════════════════════════════════════════════════════════
    # Event dispatch
    # ══════════════════════════════════════════════════════════════════════════

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

            # Store in RAG immediately
            self.agent.log_exchange(speaker, text)

            # ── Edge case: Ack during search/pending → ignore ─────────
            if (self._searching or self._pending_searches) and self._current_speaker == speaker:
                if _is_ack(text):
                    print(f"[{ts()}] 🔕 Ack during search — ignored: \"{text}\"")
                    return

            # ── Different speaker interrupts Sam ──────────────────────
            if self._speaking and self._current_speaker != speaker:
                print(f"[{ts()}] ⚡ INTERRUPT — {speaker} cut in")
                asyncio.create_task(self.speaker.stop_audio())
                self._interrupt_event.set()
                self._buffer.clear()
                self._buffer.append((speaker, text, t))
                self._restart_debounce(speaker)
                return

            # ── Same speaker adds more while Sam is speaking ──────────
            if self._speaking and self._current_speaker == speaker:
                # Don't cancel if we're in search — search continues in background
                if not self._searching:
                    if self._current_task and not self._current_task.done():
                        self._current_task.cancel()
                    if self._audio_playing:
                        asyncio.create_task(self.speaker.stop_audio())
                    self._speaking = False
                    self._audio_playing = False
                    self._interrupt_event.set()

            # Buffer the fragment
            self._buffer.append((speaker, text, t))
            self._restart_debounce(speaker)

        # ── Speech OFF → flush buffer ─────────────────────────────────────
        elif event == "participant_events.speech_off":
            speaker = payload.get("data", {}).get("data", {}).get("participant", {}).get("name", "Unknown")
            print(f"[{ts()}] 🔇 {speaker} stopped speaking")
            if self._buffer_task and not self._buffer_task.done():
                self._buffer_task.cancel()
            if self._buffer and not self._speaking:
                self._flush_buffer()

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

    # ══════════════════════════════════════════════════════════════════════════
    # Buffer + debounce
    # ══════════════════════════════════════════════════════════════════════════

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
        await self._speak_simple(greeting, t0)

    def _log_sam(self, text: str):
        self._convo_history.append(f"Sam: {text}")
        self.agent.log_exchange("Sam", text)

    def _restart_debounce(self, speaker: str):
        if self._buffer_task and not self._buffer_task.done():
            self._buffer_task.cancel()
        self._buffer_task = asyncio.create_task(self._debounce_then_flush(speaker))

    async def _debounce_then_flush(self, speaker: str):
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return
        if self._buffer and not self._speaking:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self._buffer:
            return
        speaker   = self._buffer[-1][0]
        t0        = self._buffer[0][2]
        full_text = " ".join(txt for _, txt, _ in self._buffer)
        self._buffer.clear()
        print(f"[{ts()}] 📝 Buffered complete: \"{full_text}\"")
        self._start_process(full_text, speaker, t0)

    # ══════════════════════════════════════════════════════════════════════════
    # TTS + inject helpers
    # ══════════════════════════════════════════════════════════════════════════

    async def _tts(self, text: str) -> bytes:
        async with self._tts_semaphore:
            return await self.speaker._synthesise(text)

    async def _inject_and_wait(self, audio_bytes: bytes, text: str, label: str, my_gen: int) -> bool:
        """Inject audio + interruptible playback wait. Returns False if interrupted."""
        if self._interrupt_event.is_set() or my_gen != self._generation:
            return False

        try:
            t_inj = time.time()
            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await self.speaker._inject_into_meeting(b64)
            self._audio_playing = True
            print(f"[{ts()}] ⏱ Inject {label}: {elapsed(t_inj)}")

            play_dur = max(300, len(text.split()) * 280)
            try:
                await asyncio.wait_for(self._interrupt_event.wait(), timeout=play_dur / 1000)
                print(f"[{ts()}] ⚡ Interrupted during {label}")
                self._audio_playing = False
                return False
            except asyncio.TimeoutError:
                pass
            self._audio_playing = False
            return True
        except Exception as e:
            print(f"[{ts()}] ⚠️  Inject failed ({label}): {e}")
            self._audio_playing = False
            return True

    async def _speak_simple(self, text, t0):
        """Simple TTS + inject for greetings etc."""
        if self._speaking:
            return
        self._speaking = True
        try:
            audio = await self._tts(text)
            await self._inject_and_wait(audio, text, "greeting", self._generation)
        except Exception as e:
            print(f"[{ts()}] ⚠️  _speak_simple error: {e}")
        finally:
            self._speaking = False
            self._audio_playing = False

    # ══════════════════════════════════════════════════════════════════════════
    # Background search + TTS preparation
    # ══════════════════════════════════════════════════════════════════════════

    async def _search_and_prepare_audio(self, user_text: str, context: str) -> list[tuple[str, bytes]]:
        """Background: search → summarize → TTS all sentences. Returns ready-to-inject audio."""
        summary = await self.agent.search_and_summarize(user_text, context)
        prefix = "Oh and about your earlier question."
        full = f"{prefix} {summary}"

        sentences = self.agent._split_sentences(full)
        prepared = []
        for sent in sentences:
            try:
                audio = await self._tts(sent)
                prepared.append((sent, audio))
                print(f"[{ts()}] 🔧 Pre-baked TTS: \"{sent[:50]}\"")
            except Exception as e:
                print(f"[{ts()}] ⚠️  Pre-bake TTS failed: {e}")
        return prepared

    async def _deliver_pending(self, my_gen: int):
        """Deliver all pending search results — audio pre-baked, just inject."""
        while self._pending_searches:
            if self._interrupt_event.is_set() or my_gen != self._generation:
                return

            query_text, prepare_task = self._pending_searches.pop(0)
            print(f"[{ts()}] 📬 Delivering pending: \"{query_text[:50]}\"")

            try:
                if not prepare_task.done():
                    prepared = await asyncio.wait_for(prepare_task, timeout=15)
                else:
                    prepared = prepare_task.result()
            except Exception as e:
                print(f"[{ts()}] ⚠️  Pending failed: {e}")
                continue

            if not prepared:
                continue

            full_text = " ".join(sent for sent, _ in prepared)
            for i, (sent, audio_bytes) in enumerate(prepared):
                ok = await self._inject_and_wait(audio_bytes, sent, f"pending-{i+1}", my_gen)
                if not ok:
                    return

            self._log_sam(full_text)
            self.trigger.mark_responded()
            print(f"[{ts()}] ✅ Pending delivered (0ms TTS wait)")

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
            t1 = time.time()

            # ── Trigger + Router in parallel ─────────────────────────────
            trigger_task = asyncio.create_task(
                self.trigger.should_respond(
                    text, speaker, context,
                    [e["text"] for e in self.agent.rag._entries[-20:]]
                )
            )
            router_task = asyncio.create_task(self.agent._route(text))

            print(f"[{ts()}] Trigger + Router in parallel...")
            should = await trigger_task
            print(f"[{ts()}] Trigger: {'YES' if should else 'NO'} ({elapsed(t1)})")

            if not should:
                router_task.cancel()
                return

            route = await router_task
            print(f"[{ts()}] Route: [{route}]")

            # ══════════════════════════════════════════════════════════════
            # [FT] PATH — filler + background search + TTS
            # ══════════════════════════════════════════════════════════════
            if route == "FT":
                # 1. Fire search + TTS in background immediately
                prepare_task = asyncio.create_task(
                    self._search_and_prepare_audio(text, context)
                )
                self._searching = True

                # 2. Play filler
                filler = random.choice(FILLERS)
                print(f"[{ts()}] 🗣️ Filler: \"{filler}\"")
                try:
                    filler_audio = await self._tts(filler)
                except Exception as e:
                    print(f"[{ts()}] ⚠️  Filler TTS failed: {e}")
                    # Wait for search directly
                    filler_audio = None

                if filler_audio:
                    ok = await self._inject_and_wait(filler_audio, filler, "filler", my_gen)
                    if not ok:
                        # Interrupted during filler — save search for later
                        self._pending_searches.append((text, prepare_task))
                        print(f"[{ts()}] 📥 Search saved to pending")
                        return

                # 3. Wait for search + TTS
                try:
                    prepared = await asyncio.wait_for(prepare_task, timeout=15)
                except asyncio.TimeoutError:
                    try:
                        await self._inject_and_wait(
                            await self._tts("Hmm that search is taking too long."),
                            "Hmm that search is taking too long.", "timeout", my_gen
                        )
                    except Exception:
                        pass
                    return
                except asyncio.CancelledError:
                    return

                if self._interrupt_event.is_set() or my_gen != self._generation:
                    # Interrupted after ready — save for later
                    fut = asyncio.get_event_loop().create_future()
                    fut.set_result(prepared)
                    self._pending_searches.append((text, fut))
                    print(f"[{ts()}] 📥 Search ready but interrupted — saved to pending")
                    return

                if not prepared:
                    return

                # 4. Inject pre-baked audio (0ms TTS wait)
                full_text = " ".join(sent for sent, _ in prepared)
                for i, (sent, audio_bytes) in enumerate(prepared):
                    ok = await self._inject_and_wait(audio_bytes, sent, f"search-{i+1}", my_gen)
                    if not ok:
                        break

                self._log_sam(full_text)
                self.trigger.mark_responded()
                print(f"[{ts()}] ✅ Done (search)")

            # ══════════════════════════════════════════════════════════════
            # [PM] PATH — stream LLM + sentence TTS + inject each ASAP
            # ══════════════════════════════════════════════════════════════
            else:
                sentence_queue = asyncio.Queue()
                llm_task = asyncio.create_task(
                    self.agent.stream_sentences_to_queue(text, context, sentence_queue)
                )

                all_sentences: list[str] = []
                pending_tts: list[tuple[str, asyncio.Task]] = []

                while True:
                    if self._interrupt_event.is_set() or my_gen != self._generation:
                        llm_task.cancel()
                        for _, t_task in pending_tts:
                            t_task.cancel()
                        return

                    try:
                        item = await asyncio.wait_for(sentence_queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        print(f"[{ts()}] ⚠️  LLM queue timeout")
                        break

                    if item is None:
                        break

                    if item == "__FLUSH__":
                        continue

                    all_sentences.append(item)
                    idx = len(all_sentences)
                    print(f"[{ts()}] LLM sentence {idx} ({elapsed(t1)}): \"{item}\"")
                    pending_tts.append((item, asyncio.create_task(self._tts(item))))

                    # Inject first sentence immediately
                    if idx == 1:
                        sent, task = pending_tts.pop(0)
                        try:
                            audio_bytes = await task
                            print(f"[{ts()}] ⏱ TTS sentence 1: {elapsed(t1)}")

                            ok = await self._inject_and_wait(audio_bytes, sent, "sentence-1", my_gen)
                            if ok:
                                print(f"[{ts()}] 📊 FIRST AUDIO: {elapsed(t0)}")
                            else:
                                self._log_sam(f"{' '.join(all_sentences)} [interrupted]")
                                self.trigger.mark_responded()
                                llm_task.cancel()
                                for _, t_task in pending_tts:
                                    t_task.cancel()
                                return
                        except Exception as e:
                            print(f"[{ts()}] ⚠️  TTS sentence 1 failed: {e}")

                # Inject remaining (TTS already done from prefetch)
                for i, (sent, task) in enumerate(pending_tts):
                    if self._interrupt_event.is_set() or my_gen != self._generation:
                        for _, t_task in pending_tts[i:]:
                            t_task.cancel()
                        return

                    try:
                        audio_bytes = await task
                        ok = await self._inject_and_wait(audio_bytes, sent, f"sentence-{i+2}", my_gen)
                        if not ok:
                            self._log_sam(f"{' '.join(all_sentences)} [interrupted]")
                            self.trigger.mark_responded()
                            for _, t_task in pending_tts[i+1:]:
                                t_task.cancel()
                            return
                    except Exception as e:
                        print(f"[{ts()}] ⚠️  TTS sentence {i+2} failed: {e}")

                if all_sentences:
                    self._log_sam(' '.join(all_sentences))
                    self.trigger.mark_responded()
                    print(f"[{ts()}] 📊 TOTAL: {elapsed(t0)}")
                    print(f"[{ts()}] ✅ Done (PM)")

            # ── Deliver any pending search results ───────────────────────
            await self._deliver_pending(my_gen)

        except asyncio.CancelledError:
            print(f"[{ts()}] 🔄 Task cancelled")
        except Exception as e:
            import traceback
            print(f"[{ts()}] ❌ _process error: {e}")
            traceback.print_exc()
        finally:
            self._audio_playing = False
            self._speaking      = False
            self._searching     = False

    # ══════════════════════════════════════════════════════════════════════════
    # Server start
    # ══════════════════════════════════════════════════════════════════════════

    async def start(self):
        self.agent.start()
        await self.speaker.warmup()

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        print(f"[{ts()}] WebSocket server ready on ws://0.0.0.0:{self.port}/ws")
        print(f"[{ts()}] Health check: http://localhost:{self.port}/health\n")
