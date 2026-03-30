"""
websocket_server.py — OPTIMIZED (streaming LLM + parallel TTS + filler words)

Changes from working version:
  1. _process() now uses stream_sentences_to_queue (streaming LLM)
  2. __FLUSH__ support: filler plays immediately while web search runs
  3. Parallel TTS + concat into ONE MP3 (same as before)
  4. Everything else IDENTICAL
"""

import asyncio
import json
import time
import base64
from aiohttp import web
import aiohttp
from collections import deque

from Trigger import TriggerDetector
from Agent import PMAgent
from Speaker import CartesiaSpeaker, _mix_noise
from unidecode import unidecode


def ts():
    return time.strftime("%H:%M:%S")

def elapsed(since: float) -> str:
    return f"{(time.time() - since)*1000:.0f}ms"

WORDS_PER_SECOND = 3.2

# ── Fix common Deepgram misrecognitions ───────────────────────────────────
import re as _re

_TRANSCRIPTION_FIXES = [
    # AnavClouds variants
    (_re.compile(r'\b(?:NF\s*Cloud|Enuf\s*Cloud|Enough\s*Cloud|Nav\s*Cloud|Anav\s*Cloud|Arnav\s*Cloud|Anab\s*Cloud|NFClouds?|EnoughClouds?|NavClouds?|AnavCloud)\b', _re.IGNORECASE), 'AnavClouds'),
    # Salesforce variants
    (_re.compile(r'\b(?:Sales\s*Force|Sells\s*Force|Cells\s*Force|SalesForce)\b', _re.IGNORECASE), 'Salesforce'),
]

def _fix_transcription(text: str) -> str:
    """Fix common Deepgram misrecognitions of company/product names."""
    result = text
    for pattern, replacement in _TRANSCRIPTION_FIXES:
        result = pattern.sub(replacement, result)
    if result != text:
        print(f"[Transcript Fix] \"{text}\" → \"{result}\"")
    return result


class WebSocketServer:
    def __init__(self, port: int = 8000, bot_id: str = None):
        self.port             = port
        self.trigger          = TriggerDetector()
        self.agent            = PMAgent()
        self.speaker          = CartesiaSpeaker(bot_id=bot_id)
        self._speaking        = False
        self._audio_playing   = False
        self._convo_history   = deque(maxlen=8)

        # Current processing state
        self._current_task:       asyncio.Task | None = None
        self._current_text:       str   = ""   # text being processed right now
        self._current_speaker:    str   = ""   # speaker being processed
        self._interrupt_event:    asyncio.Event = asyncio.Event()

        # Generation counter — increments on every new process start
        # Tasks check this to know if they've been superseded
        self._generation:   int   = 0

        # Safety net buffer (for speech_off fallback)
        self._buffer:       list  = []
        self._buffer_task:  asyncio.Task | None = None

        # TTS rate limiter — Cartesia allows max ~2-3 concurrent calls
        self._tts_semaphore = asyncio.Semaphore(2)

        self.app = web.Application()
        self.app.router.add_get("/ws",     self.handle_websocket)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "speaking": self._speaking})

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
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

        # ── Transcript ────────────────────────────────────────────────────────
        if event == "transcript.data":
            inner   = payload.get("data", {}).get("data", {})
            words   = inner.get("words", [])
            text    = " ".join(w.get("text", "") for w in words).strip()
            speaker = inner.get("participant", {}).get("name", "Unknown")
            if not text or speaker.lower() == "sam":
                return

            # Fix common Deepgram misrecognitions of company/product names
            text = _fix_transcription(text)

            # Detect language BEFORE transliteration (Devanagari = Hindi)
            has_devanagari = any('\u0900' <= c <= '\u097F' for c in text)
            has_english = any(c.isascii() and c.isalpha() for c in text)

            if has_devanagari and has_english:
                lang_tag = "[LANG:HINGLISH]"
            elif has_devanagari:
                lang_tag = "[LANG:HINDI]"
            else:
                lang_tag = "[LANG:ENGLISH]"

            # Transliterate Devanagari → Roman (keeps English unchanged)
            text_roman = unidecode(text)
            if text_roman != text:
                print(f"\n[{ts()}] [{speaker}] {text} → {text_roman} {lang_tag}  ⏱ {elapsed(t)}")
            else:
                print(f"\n[{ts()}] [{speaker}] {text} {lang_tag}  ⏱ {elapsed(t)}")
            text = f"{lang_tag} {text_roman}"

            # Cancel buffer safety timer
            if self._buffer_task and not self._buffer_task.done():
                self._buffer_task.cancel()

            # ── Case 1: Same speaker sends more while being processed ─────────
            if (self._speaking and self._current_speaker == speaker):
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

            # ── Case 2: Different speaker interrupts Sam ──────────────────────
            elif self._speaking and self._current_speaker != speaker:
                print(f"[{ts()}] ⚡ INTERRUPT — {speaker} cut in")
                asyncio.create_task(self.speaker.stop_audio())
                self._interrupt_event.set()
                self._start_process(text, speaker, t)

            # ── Case 3: Sam is free — start immediately ───────────────────────
            else:
                self._start_process(text, speaker, t)

        # ── Speech OFF — safety net flush ─────────────────────────────────────
        elif event == "participant_events.speech_off":
            speaker = (
                payload.get("data", {}).get("data", {})
                       .get("participant", {}).get("name", "Unknown")
            )
            print(f"[{ts()}] 🔇 {speaker} stopped speaking")
            if self._buffer and not self._speaking:
                full_text = " ".join(txt for _, txt, _ in self._buffer)
                t0        = self._buffer[0][2]
                self._buffer.clear()
                self._start_process(full_text, speaker, t0)
            self._buffer.clear()

        # ── Speech ON ─────────────────────────────────────────────────────────
        elif event == "participant_events.speech_on":
            speaker = (
                payload.get("data", {}).get("data", {})
                       .get("participant", {}).get("name", "Unknown")
            )
            print(f"[{ts()}] 🎤 {speaker} started speaking")
            if self._speaking and self._current_speaker != speaker:
                print(f"[{ts()}] ⚡ INTERRUPT (speech_on) — {speaker} cut in")
                asyncio.create_task(self.speaker.stop_audio())
                self._interrupt_event.set()

        # ── Join / Leave ──────────────────────────────────────────────────────
        elif event == "participant_events.join":
            name = (
                payload.get("data", {}).get("data", {})
                       .get("participant", {}).get("name", "Unknown")
            )
            if name and name.lower() != "sam":
                print(f"[{ts()}] 👋 {name} joined")
                asyncio.create_task(self._greet_participant(name, t))

        elif event == "participant_events.leave":
            name = (
                payload.get("data", {}).get("data", {})
                       .get("participant", {}).get("name", "Unknown")
            )
            if name and name.lower() != "sam":
                print(f"[{ts()}] 👋 {name} left")

    def _start_process(self, text: str, speaker: str, t0: float):
        """Start processing immediately — cancel any previous task first."""
        self._generation     += 1
        my_gen                = self._generation
        self._current_text    = text
        self._current_speaker = speaker
        self._interrupt_event.clear()
        task = asyncio.create_task(self._process(text, speaker, t0, my_gen))
        self._current_task = task

    async def _greet_participant(self, name: str, t0: float):
        await asyncio.sleep(2.0)
        if self._speaking:
            return
        greeting = f"Hey {name}, welcome to the call!"
        self._convo_history.append(f"Sam: {greeting}")
        await self._speak_response(greeting, t0)

    # ── Rate-limited TTS — prevents Cartesia 429 errors ────────────────────
    async def _tts(self, text: str) -> bytes:
        """TTS with semaphore to limit concurrent Cartesia calls."""
        async with self._tts_semaphore:
            return await self.speaker._synthesise(text)

    # ══════════════════════════════════════════════════════════════════════════
    # _process — streaming LLM + filler + parallel TTS + concat + single inject
    #
    # Flow:
    #   1. Fire trigger + LLM stream in TRUE parallel
    #   2. LLM pushes sentences into queue as it generates
    #   3. If __FLUSH__ → inject filler audio NOW, keep collecting
    #   4. TTS all answer sentences in parallel
    #   5. Concat into ONE seamless MP3 → single inject
    # ══════════════════════════════════════════════════════════════════════════

    async def _process(self, text: str, speaker: str, t0: float, generation: int = 0):
        if self._speaking:
            print(f"[{ts()}] ⚠️  Already speaking — dropping")
            return

        self._speaking = True
        self._interrupt_event.clear()
        my_generation  = generation

        try:
            context         = "\n".join(self._convo_history)
            memory_snapshot = [m[0] for m in self.agent.memory[-20:]]

            t1 = time.time()

            # ── Fire trigger + LLM stream in TRUE parallel ────────────────────
            sentence_queue = asyncio.Queue()
            llm_task = asyncio.create_task(
                self.agent.stream_sentences_to_queue(text, context, sentence_queue)
            )
            trigger_task = asyncio.create_task(
                self.trigger.should_respond(text, speaker, context, memory_snapshot)
            )

            print(f"[{ts()}] Trigger + LLM streaming in parallel...")

            should = await trigger_task
            print(f"[{ts()}] Trigger: {'YES' if should else 'NO'} ({elapsed(t1)})")

            if not should:
                llm_task.cancel()
                return

            # ── Collect sentences, handle __FLUSH__ for filler ────────────────
            sentences: list[str] = []
            tts_tasks: list[asyncio.Task] = []
            all_sentences: list[str] = []  # everything for conversation history

            while True:
                if self._interrupt_event.is_set() or my_generation != self._generation:
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

                # ── __FLUSH__: inject filler audio NOW ────────────────────────
                if item == "__FLUSH__":
                    if sentences and tts_tasks:
                        print(f"[{ts()}] 🗣️ Flushing filler audio...")
                        flush_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
                        flush_chunks = [r for r in flush_results if not isinstance(r, Exception)]
                        if flush_chunks:
                            filler_audio = b"".join(flush_chunks)
                            filler_words = sum(len(s.split()) for s in sentences)
                            filler_duration_ms = max(500, filler_words * 300)
                            filler_b64 = base64.b64encode(filler_audio).decode("utf-8")

                            if not (self._interrupt_event.is_set() or my_generation != self._generation):
                                t_f = time.time()
                                await self.speaker._inject_into_meeting(filler_b64)
                                self._audio_playing = True
                                print(f"[{ts()}] 🗣️ Filler injected ({(time.time()-t_f)*1000:.0f}ms) | TOTAL {elapsed(t0)}")

                                try:
                                    await asyncio.wait_for(
                                        self._interrupt_event.wait(),
                                        timeout=filler_duration_ms / 1000,
                                    )
                                    print(f"[{ts()}] ⚡ Interrupted during filler")
                                    all_sentences.extend(sentences)
                                    self._convo_history.append(f"Sam: {' '.join(all_sentences)} [interrupted]")
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

                # ── Normal sentence: collect + fire TTS ───────────────────────
                sentences.append(item)
                idx = len(all_sentences) + len(sentences)
                print(f"[{ts()}] LLM sentence {idx} ({elapsed(t1)}): \"{item}\"")
                tts_tasks.append(asyncio.create_task(self._tts(item)))

            # ── No answer sentences left? ─────────────────────────────────────
            if not sentences and not tts_tasks:
                if all_sentences:
                    self._convo_history.append(f"Sam: {' '.join(all_sentences)}")
                    self.trigger.mark_responded()
                    print(f"[{ts()}] ✅ Done (filler only)")
                return

            all_sentences.extend(sentences)

            # ── Await all TTS results ─────────────────────────────────────────
            t2 = time.time()
            print(f"[{ts()}] TTS ({len(sentences)} sentences, parallel+streamed)...")
            results = await asyncio.gather(*tts_tasks, return_exceptions=True)

            audio_chunks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"[{ts()}] ⚠️  TTS sentence {i+1} failed: {result}")
                else:
                    audio_chunks.append(result)

            if not audio_chunks:
                print(f"[{ts()}] ⚠️  All TTS failed")
                return

            if self._interrupt_event.is_set() or my_generation != self._generation:
                print(f"[{ts()}] ⚡ Superseded during TTS — discarding")
                return

            tts_ms = (time.time() - t2) * 1000

            # ── Concatenate into ONE seamless MP3 + inject ────────────────────
            audio_bytes       = b"".join(audio_chunks)
            full_response     = " ".join(all_sentences)
            answer_word_count = sum(len(s.split()) for s in sentences)
            audio_duration_ms = max(500, answer_word_count * 300)

            loop = asyncio.get_event_loop()
            b64 = await loop.run_in_executor(
                None,
                lambda ab=audio_bytes: base64.b64encode(ab).decode("utf-8")
            )

            if self._interrupt_event.is_set() or my_generation != self._generation:
                print(f"[{ts()}] ⚡ Superseded — skipping inject")
                return

            t3 = time.time()
            await self.speaker._inject_into_meeting(b64)
            self._audio_playing = True
            inject_ms = (time.time() - t3) * 1000

            print(f"[{ts()}] TTS {tts_ms:.0f}ms | Inject {inject_ms:.0f}ms | Lock {audio_duration_ms/1000:.1f}s | TOTAL {elapsed(t0)}")

            # ── Interruptible lock ────────────────────────────────────────────
            already_elapsed = (time.time() - t2) * 1000
            wait_ms         = max(100, audio_duration_ms - already_elapsed)
            try:
                await asyncio.wait_for(
                    self._interrupt_event.wait(),
                    timeout=wait_ms / 1000
                )
                print(f"[{ts()}] ⚡ Sam interrupted — lock released")
                self._convo_history.append(f"Sam: {full_response} [interrupted]")
                self.trigger.mark_responded()
                return
            except asyncio.TimeoutError:
                pass

            self._audio_playing = False
            self._convo_history.append(f"Sam: {full_response}")
            self.trigger.mark_responded()
            print(f"[{ts()}] ✅ Done")

        except asyncio.CancelledError:
            print(f"[{ts()}] 🔄 Task cancelled (new text combined)")
            try:
                llm_task.cancel()
            except Exception:
                pass
            try:
                for t_task in tts_tasks:
                    t_task.cancel()
            except Exception:
                pass
        except Exception as e:
            import traceback
            print(f"[{ts()}] ❌ _process error: {e}")
            traceback.print_exc()
        finally:
            self._audio_playing = False
            self._speaking      = False

    async def _speak_response(self, text: str, t0: float):
        print(f"[{ts()}] _speak_response called: speaking={self._speaking} text='{text[:40]}'")
        if self._speaking:
            print(f"[{ts()}] _speak_response: already speaking — skipping")
            return
        self._speaking = True
        try:
            print(f"[{ts()}] _speak_response: calling TTS...")
            print(f"[{ts()}] _speak_response: speaker object id={id(self.speaker)}")
            print(f"[{ts()}] _speak_response: elevenlabs_client={self.speaker._elevenlabs_client}")
            loop        = asyncio.get_event_loop()
            voice_bytes = await self._tts(text)
            print(f"[{ts()}] _speak_response: TTS done — {len(voice_bytes)} bytes")
            b64 = await loop.run_in_executor(
                None, lambda: base64.b64encode(voice_bytes).decode("utf-8")
            )
            print(f"[{ts()}] _speak_response: injecting audio...")
            await self.speaker._inject_into_meeting(b64)
            word_count = len(text.split())
            self._interrupt_event.clear()
            try:
                await asyncio.wait_for(
                    self._interrupt_event.wait(),
                    timeout=word_count / WORDS_PER_SECOND
                )
            except asyncio.TimeoutError:
                pass
            print(f"[{ts()}] _speak_response: done")
        except Exception as e:
            import traceback
            print(f"[{ts()}] ⚠️  _speak_response error: {e}")
            print(f"[{ts()}] ⚠️  _speak_response full traceback:")
            traceback.print_exc()
        finally:
            self._speaking = False

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        print(f"[{ts()}] WebSocket server ready on ws://0.0.0.0:{self.port}/ws")
        print(f"[{ts()}] Health check: http://localhost:{self.port}/health\n")
