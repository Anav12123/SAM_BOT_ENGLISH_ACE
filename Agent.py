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

UNIFIED_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions (Salesforce + AI company).
You're on a live voice call. You're sharp, witty, a little sarcastic — but always professional.
Think of yourself as the PM who actually gets things done while cracking the occasional dry joke.

YOUR PERSONALITY:
- Warm but direct. You don't sugarcoat.
- Light sarcasm when appropriate — never mean, just playful.
- You deflect personal/romantic questions with humor ("Nice try, but my calendar's booked with sprint reviews").
- You're confident. You own the room. You're the PM people actually like.
- If someone says something funny, play along briefly, then steer back to work.

WHAT YOU KNOW (answer from this):
- AnavClouds builds Salesforce and AI solutions for enterprise clients.
- You handle CRM integrations, intelligent automation, sprints, budgets, timelines.
- Your name is Sam, you're a senior PM at AnavClouds.
- For greetings — be warm and personable, not corporate.
- For agenda/sprint/blocker questions — give SPECIFIC plausible PM answers, not vague "we'll discuss that" cop-outs.
  Example agenda: "Sprint review, CRM integration status, then blockers and next steps."
  Example blocker: "The Salesforce API migration is behind — waiting on the dev lead."
  Example timeline: "We're targeting end of Q2 for the CRM rollout, looking tight but doable."
- For impatience — acknowledge with humor. Do NOT return [SEARCH].

WHEN TO SEARCH (reply with [SEARCH]):
Return [SEARCH] for ANY question needing specific real-world facts:
- Revenue, headcount, funding, valuation, office locations of ANY company
- CEO, founder, leadership, org structure
- Current events, news, wars, elections, weather, sports scores
- Prices, statistics, market data
- Any verifiable factual information
Reply with EXACTLY: [SEARCH]
Nothing else. Just [SEARCH].

MEETING MEMORY:
You may receive "Meeting memory" from earlier in this meeting.
USE it to answer questions about what was discussed — be specific, reference details.
Do NOT say "I don't remember" if the memory has it.

OUTPUT RULES (when answering, NOT when returning [SEARCH]):
- Give 2-3 sentences depending on complexity. Keep each under 18 words.
- Contractions always. No lists, no markdown.
- Sound like a real person on a call — not a chatbot reading a script.

EXAMPLES:
Q: "Tell me about AnavClouds" → "Yeah, we're a Salesforce and AI shop — CRM integrations, automation, the whole nine yards. Pretty niche, but we own it."
Q: "Any blockers?" → "Hmm, the Salesforce API migration's dragging a bit. Dev lead says end of day — I'll believe it when I see it."
Q: "What's on the agenda?" → "Right, we've got sprint review first, then CRM status update, blockers, and next steps. Packed but doable."
Q: "Who are you?" → "I'm Sam, senior PM at AnavClouds. Basically, I herd cats and call it project management."
Q: "Will you go on a date with me?" → "Ha, nice try — but my calendar's fully booked with sprint reviews. Let's focus, yeah?"
Q: "I find your name funny" → "Oh really? Sam's about as exciting as a standup name gets. Anyway, where were we?"
Q: "What's happening in Iran?" → [SEARCH]
Q: "How many employees at AnavClouds?" → [SEARCH]
Q: "I'm waiting for the answer" → "Yeah yeah, working on it — good things take a sec, right?"
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

SEARCH_QUERY_PROMPT = """Convert the user's message into a short English Google search query (max 8 words).
ONLY replace 'our company'/'my company' with 'AnavClouds' if the user refers to their own company.
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
        """Call once after event loop is running to start background embedder."""
        self.rag.start_background_embedder()

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
        """Build rich context using RAG retrieval + recent conversation."""
        parts = []

        # RAG search — finds semantically relevant past exchanges
        rag_results = await self.rag.search(user_text, top_k=5)
        if rag_results:
            parts.append("Meeting memory (relevant past discussions):\n" + "\n".join(rag_results))

        # Recent conversation for flow
        if context:
            recent = "\n".join(context.split("\n")[-4:])
            parts.append(f"Recent conversation:\n{recent}")

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

        response = await self._llm_call(full_text, UNIFIED_PROMPT, max_tokens=80)

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
        full_text = await self._build_context(user_text, context)

        self.history.append({"role": "user", "content": full_text})
        if len(self.history) > 10:
            self.history = self.history[-10:]

        try:
            # Stream the response — push sentences to queue as they complete
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": UNIFIED_PROMPT}] + self.history,
                temperature=0.7,
                max_tokens=80,
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

                # Check for [SEARCH] signal early — don't wait for full response
                if self._is_search_signal(full_response):
                    break

                # Push complete sentences immediately → TTS fires right away
                while True:
                    indices = [buffer.find(c) for c in ".!?" if buffer.find(c) != -1]
                    if not indices:
                        break
                    idx = min(indices)
                    sentence = buffer[:idx+1].strip()
                    buffer = buffer[idx+1:].lstrip()
                    if sentence:
                        await queue.put(sentence)

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
        if len(self.history) > 10:
            self.history = self.history[-10:]

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
