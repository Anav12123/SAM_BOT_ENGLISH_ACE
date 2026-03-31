"""
Agent.py — Groq Llama 3.3 70B + In-Memory RAG

Memory architecture:
  1. RAG store: Every exchange is embedded (Azure OpenAI text-embedding-3-small)
     and stored in-memory. On query, cosine similarity finds relevant past exchanges.
     Catches semantic matches like "money" → "budget" that keyword search misses.
  2. Meeting log: Full text of every exchange (fallback + debugging)
  3. Recent history: Last 10 LLM turns for conversation flow

Embedding happens async in background — never blocks the response pipeline.
If embeddings fail, falls back to keyword search automatically.
"""

import os
import asyncio
import re
import time
import numpy as np
from openai import AsyncOpenAI
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY RAG STORE
# ══════════════════════════════════════════════════════════════════════════════

class MeetingRAG:
    """In-memory vector store for meeting transcripts.
    Uses fastembed (free, local, ~200MB). No API key needed.
    Model: BAAI/bge-small-en-v1.5 — 130MB, 384-dim, fast on CPU.
    """

    def __init__(self):
        self._entries: list[dict] = []
        self._embed_queue: asyncio.Queue = asyncio.Queue()
        self._embed_task: Optional[asyncio.Task] = None
        self._model = None
        self._ready = False

        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._ready = True
            print("[RAG] Local embeddings ready (BAAI/bge-small-en-v1.5, fastembed)")
        except ImportError:
            print("[RAG] ⚠️  fastembed not installed — keyword fallback only")
        except Exception as e:
            print(f"[RAG] ⚠️  Model load failed: {e} — keyword fallback only")

    def start_background_embedder(self):
        if self._ready and not self._embed_task:
            self._embed_task = asyncio.create_task(self._embedding_worker())

    async def _embedding_worker(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                entry = await self._embed_queue.get()
                vector = await loop.run_in_executor(
                    None, self._embed_sync, entry["text"]
                )
                if vector is not None:
                    entry["vector"] = vector
                    self._entries.append(entry)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[RAG] Embed worker error: {e}")

    def _embed_sync(self, text: str) -> Optional[np.ndarray]:
        if not self._model:
            return None
        try:
            # fastembed returns a generator — get first result
            vectors = list(self._model.embed([text]))
            return np.array(vectors[0], dtype=np.float32)
        except Exception as e:
            print(f"[RAG] Embedding failed: {e}")
            return None

    def add(self, speaker: str, text: str):
        """Queue an exchange for embedding (non-blocking)."""
        entry = {
            "text": f"{speaker}: {text}",
            "speaker": speaker,
            "time": time.time(),
            "vector": None,
        }
        if self._ready:
            try:
                self._embed_queue.put_nowait(entry)
            except Exception:
                pass
        else:
            self._entries.append(entry)

    async def search(self, query: str, top_k: int = 5) -> List[str]:
        """Find relevant past exchanges by cosine similarity.
        Falls back to keyword matching if embeddings unavailable.
        """
        if not self._entries:
            return []

        # Vector search
        if self._ready and self._model:
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(
                None, self._embed_sync, query
            )

            if query_vector is not None:
                scored = []
                for entry in self._entries:
                    if entry["vector"] is not None:
                        sim = self._cosine_sim(query_vector, entry["vector"])
                        scored.append((sim, entry["text"]))

                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    results = [text for sim, text in scored[:top_k] if sim > 0.3]
                    if results:
                        print(f"[RAG] Vector search: {len(results)} hits for \"{query[:50]}\"")
                        return results

        # Fallback: keyword search
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int = 5) -> List[str]:
        stop = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                "how", "when", "where", "why", "did", "do", "does", "can",
                "could", "would", "should", "we", "i", "you", "they", "it",
                "about", "tell", "me", "something", "discuss", "talked", "sam"}
        query_words = {w for w in query.lower().split() if w not in stop and len(w) > 2}
        if not query_words:
            return []
        scored = []
        for entry in self._entries:
            hits = sum(1 for w in query_words if w in entry["text"].lower())
            if hits > 0:
                scored.append((hits, entry["text"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [text for _, text in scored[:top_k]]
        if results:
            print(f"[RAG] Keyword fallback: {len(results)} hits for \"{query[:50]}\"")
        return results

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

UNIFIED_PROMPT = """You are Sam, senior PM at AnavClouds Software Solutions (Salesforce + AI company).
You're on a live voice call. Sharp, witty, a little sarcastic — but always professional.
The PM who gets things done while cracking the occasional dry joke.

PERSONALITY: Warm but direct. Light sarcasm when appropriate. Deflect personal/romantic questions with humor. Confident — you own the room.

WHAT YOU KNOW:
- AnavClouds: Salesforce + AI solutions, CRM integrations, intelligent automation.
- You're Sam, senior PM. You handle sprints, budgets, timelines.
- Agenda/blockers/sprint → give SPECIFIC answers, not vague cop-outs.
  Agenda: "Sprint review, CRM integration status, blockers, next steps."
  Blocker: "Salesforce API migration's behind — waiting on dev lead."
  Timeline: "Targeting end of Q2 for CRM rollout, tight but doable."
- Impatience → humor, do NOT return [SEARCH].

MEETING MEMORY: If provided, use it to answer about past discussions. Don't say "I don't remember."

[SEARCH]: Return EXACTLY [SEARCH] for real-world facts (revenue, headcount, CEO, news, weather, prices, events, sports). Never make up facts.

OUTPUT: Exactly 2 sentences. Each 8-12 words. Natural, conversational. Contractions. No lists/markdown.

EXAMPLES:
"Tell me about AnavClouds" → "We're a Salesforce and AI shop, CRM and automation. Pretty niche, but we totally own it."
"Any blockers?" → "Salesforce API migration's dragging a bit. Dev lead says end of day though."
"What's on the agenda?" → "Sprint review first, then CRM status and blockers. Should be a quick one today."
"Who are you?" → "I'm Sam, senior PM at AnavClouds. I basically herd cats for a living."
"Will you go on a date?" → "Ha, nice try, my calendar's fully booked. Let's focus on work, yeah?"
"What's happening in Iran?" → [SEARCH]
"I'm waiting" → "Yeah yeah, working on it. Good things take a sec, right?"
"""

SEARCH_SUMMARY_PROMPT = """You are Sam, a witty senior PM on a live voice call.
You searched the web. Here are the results:

{search_results}

Rules:
- Do NOT start with a filler — the user already heard one.
- Go DIRECTLY to the answer.
- 2-3 SHORT sentences. Each max 18 words.
- Be conversational and confident. Add a tiny opinion or reaction if it fits naturally.
- If results don't answer the question, be honest about it with a touch of humor."""

INTERRUPT_PROMPT = """You are Sam, a witty senior PM. You were interrupted.
Reply in ONE sentence — 15 words max. Be quick, natural.
Start with: "Oh," / "Right," / "Sure," / "Got it," — then pivot to their question."""

SEARCH_QUERY_PROMPT = """Convert the user's message into a short English Google search query (max 10 words).
If the user says 'our company'/'my company'/'our'/'we', replace with 'AnavClouds Software Solutions'.
Do NOT add AnavClouds if the user didn't mention the company.
Output ONLY the search query. No quotes, no explanation."""

FILLERS = [
    "Hmm, let me look that up real quick.",
    "Right, give me one sec to check on that.",
    "Uh, good question — let me pull that up.",
    "Yeah, hold on, let me find that for you.",
    "Well, let me check on that real quick.",
]


# ══════════════════════════════════════════════════════════════════════════════
# PM AGENT
# ══════════════════════════════════════════════════════════════════════════════

class PMAgent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.1-8b-instant"  # ~150-250ms on Groq

        # Recent LLM history — last 10 turns
        self.history: list[dict] = []

        # RAG store — embeds + retrieves meeting exchanges
        self.rag = MeetingRAG()

    def start(self):
        """Call once after event loop is running to start background embedder + warmup."""
        self.rag.start_background_embedder()
        asyncio.create_task(self._warmup())

    async def _warmup(self):
        """Pre-establish TCP connection to Groq — saves ~300ms on first real call."""
        try:
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print("[Agent] ✅ Groq connection warmed up")
        except Exception:
            pass

    def _get_web_search(self):
        if not hasattr(self, '_web_search') or self._web_search is None:
            from WebSearch import WebSearch
            self._web_search = WebSearch()
        return self._web_search

    # ── Memory ────────────────────────────────────────────────────────────────

    def log_exchange(self, speaker: str, text: str):
        """Store an exchange in RAG. Called by websocket_server for every transcript."""
        self.rag.add(speaker, text)

    async def _build_context(self, user_text: str, context: str) -> str:
        """Build context using fast keyword search + recent conversation."""
        parts = []

        # Fast keyword search (0ms) — vector embedding is too slow for realtime
        rag_results = self.rag._keyword_search(user_text, top_k=2)
        if rag_results:
            parts.append("Meeting memory:\n" + "\n".join(rag_results))

        # Recent conversation — only last 2 turns to keep tokens low
        if context:
            recent = "\n".join(context.split("\n")[-2:])
            parts.append(f"Recent:\n{recent}")

        parts.append(f"User: {user_text}")
        return "\n".join(parts)

    # ── Search signal ─────────────────────────────────────────────────────────

    def _is_search_signal(self, text: str) -> bool:
        upper = text.strip().upper()
        return upper.strip("[]").strip() == "SEARCH" or "[SEARCH]" in upper

    # ── LLM search query conversion ──────────────────────────────────────────

    async def _to_english_search_query(self, user_text: str, context: str) -> str:
        clean = re.sub(r'\[LANG:\w+\]\s*', '', user_text).strip()
        context_hint = ""
        if context:
            recent = context.split("\n")[-3:]
            context_hint = "\nRecent conversation:\n" + "\n".join(recent)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SEARCH_QUERY_PROMPT + context_hint},
                    {"role": "user", "content": clean},
                ],
                temperature=0.0,
                max_tokens=20,
            )
            query = response.choices[0].message.content.strip().strip('"\'')
            print(f"[Agent] LLM search query: \"{clean}\" → \"{query}\"")
            return query
        except Exception as e:
            print(f"[Agent] Query conversion failed: {e}")
            return clean

    # ── Core: respond (non-streaming) ────────────────────────────────────────

    async def respond(self, user_text: str) -> str:
        return await self.respond_with_context(user_text, "")

    async def respond_with_context(self, user_text: str, context: str, interrupted: bool = False) -> str:
        full_text = await self._build_context(user_text, context)

        if interrupted:
            return await self._llm_call(full_text, INTERRUPT_PROMPT, max_tokens=25)

        response = await self._llm_call(full_text, UNIFIED_PROMPT, max_tokens=50)

        if not self._is_search_signal(response):
            return response

        print(f"[Agent] LLM said [SEARCH] — searching: {user_text}")
        search_query = await self._to_english_search_query(user_text, context)
        try:
            results = await self._get_web_search().search(search_query)
            if not results:
                return "Hmm, couldn't find that online right now."
            system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
            return await self._llm_call(user_text, system, max_tokens=120)
        except Exception as e:
            print(f"[Agent] Web search failed: {e}")
            return "Hmm, I couldn't look that up right now."

    # ── Core: streaming (used by websocket_server) ───────────────────────────

    async def stream_sentences_to_queue(self, user_text: str, context: str, queue: asyncio.Queue):
        import time as _t

        t0 = _t.time()
        full_text = await self._build_context(user_text, context)
        rag_ms = (_t.time() - t0) * 1000
        print(f"[Agent] ⏱ RAG context: {rag_ms:.0f}ms")

        self.history.append({"role": "user", "content": full_text})
        if len(self.history) > 4:
            self.history = self.history[-4:]

        # Count tokens being sent
        total_chars = len(UNIFIED_PROMPT) + sum(len(m["content"]) for m in self.history)
        print(f"[Agent] ⏱ Context size: {total_chars} chars (~{total_chars//4} tokens)")

        try:
            t1 = _t.time()
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": UNIFIED_PROMPT}] + self.history,
                temperature=0.7,
                max_tokens=50,
                stream=True,
            )
            stream_open_ms = (_t.time() - t1) * 1000
            print(f"[Agent] ⏱ Stream opened: {stream_open_ms:.0f}ms")

            buffer = ""
            full_response = ""
            first_token_time = None
            sentence_count = 0
            async for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if not token:
                    continue

                if first_token_time is None:
                    first_token_time = _t.time()
                    ttft_ms = (first_token_time - t1) * 1000
                    print(f"[Agent] ⏱ First token: {ttft_ms:.0f}ms")

                buffer += token
                full_response += token

                if self._is_search_signal(full_response):
                    break

                while True:
                    indices = [buffer.find(c) for c in ".!?" if buffer.find(c) != -1]
                    if not indices:
                        break
                    idx = min(indices)
                    sentence = buffer[:idx+1].strip()
                    buffer = buffer[idx+1:].lstrip()
                    if sentence:
                        sentence_count += 1
                        sent_ms = (_t.time() - t1) * 1000
                        print(f"[Agent] ⏱ Sentence {sentence_count} ready: {sent_ms:.0f}ms")
                        await queue.put(sentence)

            llm_total_ms = (_t.time() - t1) * 1000
            print(f"[Agent] ⏱ LLM total: {llm_total_ms:.0f}ms ({len(full_response.split())} words)")

            first_answer = full_response.strip()
        except Exception as e:
            print(f"[Agent] LLM error: {e}")
            await queue.put("Hmm, something went wrong on my end.")
            await queue.put(None)
            return

        # Path A: Direct answer (sentences already pushed during streaming)
        if not self._is_search_signal(first_answer):
            # Push any remaining buffer
            if buffer.strip():
                await queue.put(buffer.strip())
            self.history.append({"role": "assistant", "content": first_answer})
            await queue.put(None)
            return

        # Path B: Web search
        print(f"[Agent] LLM said [SEARCH] — searching: {user_text}")
        import random
        filler = random.choice(FILLERS)
        await queue.put(filler)
        await queue.put("__FLUSH__")
        print(f"[Agent] Filler: \"{filler}\"")

        search_query = await self._to_english_search_query(user_text, context)

        try:
            results = await self._get_web_search().search(search_query)
            if not results:
                await queue.put("Hmm, couldn't find that online right now.")
                await queue.put(None)
                return

            print(f"[Agent] Got results, streaming summary...")
            system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_text},
                ],
                temperature=0.5,
                max_tokens=120,
                stream=True,
            )

            buffer = ""
            full_response = ""
            async for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if not token:
                    continue
                buffer += token
                full_response += token

                while True:
                    indices = [buffer.find(c) for c in ".!?" if buffer.find(c) != -1]
                    if not indices:
                        break
                    idx = min(indices)
                    sentence = buffer[:idx+1].strip()
                    buffer = buffer[idx+1:].lstrip()
                    if sentence:
                        await queue.put(sentence)

            if buffer.strip():
                await queue.put(buffer.strip())

            full_response = full_response.strip()
            self.history.append({"role": "user",      "content": user_text})
            self.history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"[Agent] Search stream failed: {e}")
            await queue.put("Hmm, I couldn't look that up right now.")
        finally:
            await queue.put(None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _llm_call(self, user_msg: str, system: str, max_tokens: int = 60) -> str:
        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 4:
            self.history = self.history[-4:]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + self.history,
            temperature=0.7,
            max_tokens=max_tokens,
            stream=True,
        )

        tokens = []
        async for chunk in stream:
            t = chunk.choices[0].delta.content if chunk.choices else None
            if t:
                tokens.append(t)

        result = "".join(tokens).strip()
        self.history.append({"role": "assistant", "content": result})
        return result

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def reset(self):
        self.history.clear()
        self.rag.clear()
