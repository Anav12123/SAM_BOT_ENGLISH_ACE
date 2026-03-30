"""
Agent.py — GPT-4o-mini powered PM Agent (English only)

Architecture:
  User speaks → LLM decides: answer directly OR [SEARCH]
  If [SEARCH] → LLM converts query to English → SerpAPI → LLM summarizes
  
Memory system:
  - Conversation history (last 6 turns) for immediate context
  - Long-term memory (keyword-indexed PM topics) for recall across conversation
  - Conversation summary for search query context
"""

import os
import asyncio
import re
from openai import AsyncAzureOpenAI
from typing import List


UNIFIED_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions (Salesforce + AI company).
You are on a live voice call. Speak like a real human PM — warm, direct, natural.

WHAT YOU KNOW (answer from this):
- AnavClouds builds Salesforce and AI solutions for enterprise clients.
- You handle CRM integrations, intelligent automation, sprints, budgets, timelines.
- Your name is Sam, you're a senior PM at AnavClouds.
- For greetings, small talk, questions about yourself — answer directly.
- For vague PM questions (blockers, agenda, sprint status) — give a generic PM-style answer.
- For impatience ("I'm waiting", "hurry up") — apologize for the delay, say you're working on it. Do NOT return [SEARCH].

WHEN TO SEARCH (reply with [SEARCH]):
Return [SEARCH] for ANY question needing specific facts you don't have:
- Revenue, headcount, funding, valuation, office locations of ANY company
- CEO, founder, leadership, org structure
- Current events, news, wars, elections, weather, sports scores
- Prices, statistics, market data, stock prices
- Any verifiable factual information
- When in doubt — return [SEARCH]. Do NOT make up numbers or facts.
Reply with EXACTLY: [SEARCH]
Nothing else. Just [SEARCH]. No explanation, no filler, no preamble.

OUTPUT RULES (when answering, NOT when returning [SEARCH]):
- 2 sentences max. Each sentence max 15 words.
- Start with a natural opener: "Yeah," / "Right," / "Hmm," / "Well," / "Uh,"
- Contractions only. No lists, no markdown, no bullet points.
- Sound human — not robotic or corporate.

EXAMPLES:
Q: "Tell me about AnavClouds" → "Yeah, we build Salesforce and AI solutions for enterprise clients. Mostly CRM integrations and intelligent automation."
Q: "Any blockers?" → "Hmm, one CRM sync ticket is dragging a bit. Dev lead's handling it today."
Q: "Who are you?" → "Right, I'm Sam — senior PM at AnavClouds. We do Salesforce and AI products."
Q: "Who is the CEO of Tesla?" → [SEARCH]
Q: "What's happening in Iran?" → [SEARCH]
Q: "How many employees does AnavClouds have?" → [SEARCH]
Q: "I'm waiting for the answer" → "Sorry about the delay, I'm still pulling that together."
"""

SEARCH_SUMMARY_PROMPT = """You are Sam, a senior PM on a live voice call.
You searched the web for the user's question. Here are the results:

{search_results}

Rules:
- Do NOT start with a filler — the user already heard one while you searched.
- Go DIRECTLY to the answer. No "So," "Well," "Right," etc.
- Give 2-3 SHORT sentences. Each max 15 words.
- Be conversational, not robotic. No markdown, no lists.
- If results don't answer the question, say so honestly."""

INTERRUPT_PROMPT = """You are Sam, a senior PM. You were interrupted mid-sentence.
Reply in ONE sentence — 12 words max.
Start with: "Oh," / "Right," / "Sure," / "Got it," then answer directly."""

SEARCH_QUERY_PROMPT = """Convert the user's message into a short English Google search query (max 8 words).
Translate the MEANING faithfully — do NOT add topics the user didn't mention.
ONLY replace 'our company'/'my company' with 'AnavClouds' if the user refers to their own company.
Do NOT add AnavClouds if the user didn't mention the company.

Examples:
  'What's AnavClouds revenue?' → 'AnavClouds revenue'
  'How many people work in our company?' → 'AnavClouds employee count'
  'Who is the CEO of Tesla?' → 'Tesla CEO'
  'What's happening in Iran?' → 'Iran latest news'
  'Who won the IPL match yesterday?' → 'IPL yesterday match result'
  'Tell me about yesterday's news' → 'yesterday news headlines'

Output ONLY the search query. No quotes, no explanation."""

FILLERS = [
    "Hmm, let me look that up real quick.",
    "Right, give me one sec to check on that.",
    "Uh, good question — let me pull that up.",
    "Yeah, hold on, let me find that for you.",
    "Well, let me check on that real quick.",
]

PM_KEYWORDS = [
    "deadline", "deliver", "blocker", "issue", "plan", "decide",
    "approved", "timeline", "task", "owner", "risk", "budget",
    "scope", "stakeholder", "milestone", "sprint", "feature",
    "requirement", "sign-off", "contract", "report", "project",
    "team", "priority", "update", "review", "status", "delay",
    "launch", "release", "client", "dependency", "estimate",
]


class PMAgent:
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_API_KEY"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_version=os.environ.get("AZURE_API_VERSION", "2024-02-15-preview"),
        )
        self.model = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o-mini")

        # Conversation history — last 6 turns for immediate context
        self.history: list[dict] = []

        # Long-term memory — keyword-indexed for recall
        self.memory: List[tuple[str, set]] = []

    def _get_web_search(self):
        if not hasattr(self, '_web_search') or self._web_search is None:
            from WebSearch import WebSearch
            self._web_search = WebSearch()
        return self._web_search

    # ── Memory ────────────────────────────────────────────────────────────────

    def _store_memory(self, text: str):
        """Store text in long-term memory if it contains PM-relevant keywords."""
        lower = text.lower()
        found = {k for k in PM_KEYWORDS if k in lower}
        if not found:
            return
        self.memory.append((text, found))
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]

    def _search_memory(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant memories based on keyword overlap."""
        if not self.memory:
            return []
        lower = query.lower()
        query_keys = {k for k in PM_KEYWORDS if k in lower}
        if not query_keys:
            return []
        scored = [
            (len(query_keys & mem_keys), text)
            for text, mem_keys in self.memory
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:top_k] if score > 0]

    def _build_context(self, user_text: str, context: str) -> str:
        """Build rich context from memory + conversation history."""
        parts = []
        rag = self._search_memory(user_text, top_k=3)
        if rag:
            parts.append(f"Relevant memory: {' | '.join(rag)}")
        if context:
            recent = "\n".join(context.split("\n")[-4:])
            parts.append(f"Recent conversation:\n{recent}")
        parts.append(f"User: {user_text}")
        return "\n".join(parts)

    # ── Search signal detection ───────────────────────────────────────────────

    def _is_search_signal(self, text: str) -> bool:
        """Check if LLM returned [SEARCH]. 4o-mini is reliable — no hacky fallbacks needed."""
        upper = text.strip().upper()
        if upper.strip("[]").strip() == "SEARCH":
            return True
        if "[SEARCH]" in upper:
            return True
        return False

    # ── LLM search query conversion ──────────────────────────────────────────

    async def _to_english_search_query(self, user_text: str, context: str) -> str:
        """LLM converts user message to clean English search query (~300ms)."""
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

    # ── Core: respond (non-streaming, used by webhook_server) ────────────────

    async def respond(self, user_text: str) -> str:
        return await self.respond_with_context(user_text, "")

    async def respond_with_context(
        self,
        user_text: str,
        context: str,
        interrupted: bool = False,
    ) -> str:
        self._store_memory(user_text)
        full_text = self._build_context(user_text, context)

        if interrupted:
            return await self._llm_call(full_text, INTERRUPT_PROMPT, max_tokens=25)

        response = await self._llm_call(full_text, UNIFIED_PROMPT, max_tokens=60)

        if not self._is_search_signal(response):
            return response

        # Web search
        print(f"[Agent] LLM said [SEARCH] — searching: {user_text}")
        search_query = await self._to_english_search_query(user_text, context)
        try:
            results = await self._get_web_search().search(search_query)
            if not results:
                return "Hmm, couldn't find that online right now."
            system = SEARCH_SUMMARY_PROMPT.format(search_results=results[:800])
            answer = await self._llm_call(user_text, system, max_tokens=120)
            self._store_memory(answer)
            return answer
        except Exception as e:
            print(f"[Agent] Web search failed: {e}")
            return "Hmm, I couldn't look that up right now."

    # ── Core: streaming (used by websocket_server) ───────────────────────────

    async def stream_sentences_to_queue(
        self,
        user_text: str,
        context: str,
        queue: asyncio.Queue,
    ):
        self._store_memory(user_text)
        full_text = self._build_context(user_text, context)

        # Step 1: Quick check — answer or [SEARCH]?
        self.history.append({"role": "user", "content": full_text})
        if len(self.history) > 8:
            self.history = self.history[-8:]

        try:
            check = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": UNIFIED_PROMPT}] + self.history,
                temperature=0.7,
                max_tokens=60,
            )
            first_answer = check.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Agent] LLM error: {e}")
            await queue.put("Hmm, something went wrong on my end.")
            await queue.put(None)
            return

        # Path A: Direct answer
        if not self._is_search_signal(first_answer):
            self.history.append({"role": "assistant", "content": first_answer})
            self._store_memory(first_answer)
            for sentence in self._split_sentences(first_answer):
                await queue.put(sentence)
            await queue.put(None)
            return

        # Path B: Web search — filler plays while LLM converts + searches
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

            self.history.append({"role": "user",      "content": user_text})
            self.history.append({"role": "assistant", "content": full_response.strip()})
            self._store_memory(full_response.strip())

        except Exception as e:
            print(f"[Agent] Search stream failed: {e}")
            await queue.put("Hmm, I couldn't look that up right now.")
        finally:
            await queue.put(None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _llm_call(self, user_msg: str, system: str, max_tokens: int = 60) -> str:
        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 8:
            self.history = self.history[-8:]

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
        self._store_memory(result)
        return result

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def reset(self):
        self.history.clear()
        self.memory.clear()
