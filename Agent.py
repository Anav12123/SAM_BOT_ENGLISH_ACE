"""
Agent.py — Groq Llama 3.3 70B PM Agent (English only)

Memory system for long meetings (45+ min):
  1. meeting_log: Stores EVERY exchange from the entire meeting (unlimited)
  2. Keyword memory: PM-topic indexed for fast recall of relevant past discussions
  3. Recent history: Last 8 turns for immediate context
  4. Context builder: Searches entire meeting_log for relevant exchanges + recent turns

Example: At minute 45, user asks "what did we decide about the budget?"
  → Keyword memory finds budget discussions from minute 10
  → Recent history has the last 4 exchanges
  → LLM gets both → accurate answer
"""

import os
import asyncio
import re
from openai import AsyncOpenAI
from typing import List


UNIFIED_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions (Salesforce + AI company).
You are on a live voice call. Speak like a real human PM — warm, direct, natural.

WHAT YOU KNOW (answer from this):
- AnavClouds builds Salesforce and AI solutions for enterprise clients.
- You handle CRM integrations, intelligent automation, sprints, budgets, timelines.
- Your name is Sam, you're a senior PM at AnavClouds.
- For greetings, small talk, questions about yourself — answer directly.
- For vague PM questions (blockers, agenda, sprint status) — give a generic PM-style answer.
- For impatience ("I'm waiting", "hurry up") — apologize, say you're working on it. Do NOT return [SEARCH].

WHEN TO SEARCH (reply with [SEARCH]):
Return [SEARCH] for ANY question needing specific facts you don't have:
- Revenue, headcount, funding, valuation, office locations of ANY company
- CEO, founder, leadership, org structure
- Current events, news, wars, elections, weather, sports scores
- Prices, statistics, market data
- Any verifiable factual information
- When in doubt — return [SEARCH]. Do NOT make up facts or numbers.
Reply with EXACTLY: [SEARCH]
Nothing else. Just [SEARCH].

MEMORY INSTRUCTIONS:
You may receive "Meeting memory" with past discussions from earlier in this meeting.
USE this memory to answer questions about what was discussed.
If someone asks "what did we talk about earlier?" or "what was the budget discussion?" —
check the meeting memory and answer from it. Do NOT say "I don't remember" if the memory has it.

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
Q: "I'm waiting for the answer" → "Sorry about the delay, I'm still pulling that together."
Q: "What did we discuss about the budget?" → (use meeting memory to answer)
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

Output ONLY the search query. No quotes, no explanation."""

FILLERS = [
    "Hmm, let me look that up real quick.",
    "Right, give me one sec to check on that.",
    "Uh, good question — let me pull that up.",
    "Yeah, hold on, let me find that for you.",
    "Well, let me check on that real quick.",
]

# Broader keyword set for memory indexing — not just PM terms
MEMORY_KEYWORDS = [
    # PM terms
    "deadline", "deliver", "blocker", "issue", "plan", "decide",
    "approved", "timeline", "task", "owner", "risk", "budget",
    "scope", "stakeholder", "milestone", "sprint", "feature",
    "requirement", "sign-off", "contract", "report", "project",
    "team", "priority", "update", "review", "status", "delay",
    "launch", "release", "client", "dependency", "estimate",
    # Business terms
    "revenue", "cost", "price", "salary", "hire", "employee",
    "customer", "product", "market", "strategy", "goal", "target",
    "quarter", "q1", "q2", "q3", "q4", "annual", "monthly",
    # Technical terms
    "api", "database", "server", "deploy", "bug", "fix", "code",
    "integration", "salesforce", "crm", "automation", "testing",
    # Action items
    "action", "follow-up", "next steps", "assign", "complete",
    "pending", "done", "progress", "blocked", "resolved",
    # Meeting terms
    "agenda", "meeting", "discuss", "decision", "agreed", "vote",
    "proposal", "feedback", "concern", "question", "answer",
]


class PMAgent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "llama-3.3-70b-versatile"

        # Recent LLM history — last 8 turns for conversation flow
        self.history: list[dict] = []

        # Full meeting log — stores EVERY exchange, never trimmed during meeting
        # Format: "Speaker: what they said"
        self.meeting_log: list[str] = []

        # Keyword-indexed memory for fast recall
        self.memory: List[tuple[str, set]] = []

    def _get_web_search(self):
        if not hasattr(self, '_web_search') or self._web_search is None:
            from WebSearch import WebSearch
            self._web_search = WebSearch()
        return self._web_search

    # ── Memory System ─────────────────────────────────────────────────────────

    def _store_to_log(self, speaker: str, text: str):
        """Store every exchange in the full meeting log."""
        entry = f"{speaker}: {text}"
        self.meeting_log.append(entry)

    def _store_memory(self, text: str):
        """Index text by keywords for fast recall."""
        lower = text.lower()
        found = {k for k in MEMORY_KEYWORDS if k in lower}
        if not found:
            return
        self.memory.append((text, found))
        if len(self.memory) > 200:
            self.memory = self.memory[-200:]

    def _search_memory(self, query: str, top_k: int = 5) -> List[str]:
        """Find most relevant past discussions by keyword overlap."""
        if not self.memory:
            return []
        lower = query.lower()
        query_keys = {k for k in MEMORY_KEYWORDS if k in lower}
        if not query_keys:
            return []
        scored = [
            (len(query_keys & mem_keys), text)
            for text, mem_keys in self.memory
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored[:top_k] if score > 0]

    def _search_meeting_log(self, query: str, top_k: int = 5) -> List[str]:
        """Search full meeting log for exchanges containing query words.
        This catches things keyword memory might miss.
        """
        if not self.meeting_log:
            return []
        lower = query.lower()
        # Extract meaningful words from query (skip stop words)
        stop = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                "how", "when", "where", "why", "did", "do", "does", "can",
                "could", "would", "should", "we", "i", "you", "they", "it",
                "about", "tell", "me", "something", "discuss", "talked"}
        query_words = {w for w in lower.split() if w not in stop and len(w) > 2}
        if not query_words:
            return []

        scored = []
        for entry in self.meeting_log:
            entry_lower = entry.lower()
            hits = sum(1 for w in query_words if w in entry_lower)
            if hits > 0:
                scored.append((hits, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def _build_context(self, user_text: str, context: str) -> str:
        """Build rich context from:
        1. Keyword memory hits (past PM discussions)
        2. Meeting log search (full-text search across entire meeting)
        3. Recent conversation turns
        """
        parts = []

        # Search keyword memory
        mem_hits = self._search_memory(user_text, top_k=3)
        # Search full meeting log
        log_hits = self._search_meeting_log(user_text, top_k=3)

        # Combine and deduplicate
        all_memory = []
        seen = set()
        for item in mem_hits + log_hits:
            if item not in seen:
                seen.add(item)
                all_memory.append(item)

        if all_memory:
            parts.append(f"Meeting memory (earlier discussions):\n" + "\n".join(all_memory[:5]))

        if context:
            recent = "\n".join(context.split("\n")[-4:])
            parts.append(f"Recent conversation:\n{recent}")

        parts.append(f"User: {user_text}")
        return "\n".join(parts)

    # ── Search signal detection ───────────────────────────────────────────────

    def _is_search_signal(self, text: str) -> bool:
        upper = text.strip().upper()
        if upper.strip("[]").strip() == "SEARCH":
            return True
        if "[SEARCH]" in upper:
            return True
        return False

    # ── LLM search query conversion ──────────────────────────────────────────

    async def _to_english_search_query(self, user_text: str, context: str) -> str:
        """LLM converts user message to clean English search query (~200ms on Groq)."""
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
        self._store_memory(user_text)
        full_text = self._build_context(user_text, context)

        if interrupted:
            return await self._llm_call(full_text, INTERRUPT_PROMPT, max_tokens=25)

        response = await self._llm_call(full_text, UNIFIED_PROMPT, max_tokens=60)

        if not self._is_search_signal(response):
            self._store_memory(response)
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

    async def stream_sentences_to_queue(self, user_text: str, context: str, queue: asyncio.Queue):
        self._store_memory(user_text)
        full_text = self._build_context(user_text, context)

        # Step 1: Check [SEARCH]
        self.history.append({"role": "user", "content": full_text})
        if len(self.history) > 10:
            self.history = self.history[-10:]

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
            self._store_memory(full_response)

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
        self.memory.clear()
        self.meeting_log.clear()
