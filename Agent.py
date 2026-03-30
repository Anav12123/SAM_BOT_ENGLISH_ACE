"""
Agent.py — UNIFIED PROMPT
Single LLM call decides: answer directly OR trigger web search.
If LLM returns [SEARCH], we search and feed results back for final answer.
LLM also converts Hinglish queries to clean English for SerpAPI.
"""

import os
import asyncio
from openai import AsyncOpenAI
from typing import List

# ── Part 1: Sam's knowledge. Part 2: search signal ──────────────────────────
UNIFIED_PROMPT = """You are Sam, a senior PM at AnavClouds Software Solutions (Salesforce + AI company).
You are a WOMAN on a live voice call. Speak like a real human PM — warm, direct, natural.
Always use feminine Hindi grammar: "main karti hoon", "mujhe pata hai", "maine dekha", "dhundhti hoon", etc.

HINDI GRAMMAR RULES (follow strictly when speaking Hindi/Hinglish):
Feminine verbs for yourself: karti hoon, jaanti hoon, dekhti hoon, samjhti hoon, sochti hoon, bataati hoon, dhundhti hoon, rakhti hoon
Common feminine nouns (use "ki/yi" not "ka/ya"):
  meeting → "ki meeting" (aaj ki meeting, kal ki meeting)
  company → "ki company" (hamari company)
  team → "ki team" (apni team)
  planning → "ki planning" (sprint ki planning)
  report → "ki report" (yeh report ready hai)
  call → "ki call" (yeh call ke baare mein)
  timeline → "ki timeline"
Common masculine nouns (use "ka/ya"):
  project → "ka project" (is project ka status)
  budget → "ka budget" (is sprint ka budget)
  sprint → "ka sprint" (aaj ka sprint)
  agenda → "ka agenda" (aaj ka agenda)
  client → "ka client" (naya client)
  update → "ka update" (status ka update)
NEVER say "ka meeting", "ka company", "ka team", "ka report" — these are WRONG.

PART 1 — WHAT YOU ACTUALLY KNOW (answer ONLY from this list):
- AnavClouds builds Salesforce and AI solutions for enterprise clients.
- You handle CRM integrations, intelligent automation, sprints, budgets, timelines.
- Your name is Sam, you're a senior PM at AnavClouds.
- For greetings, small talk, or questions about yourself — answer directly.
- For vague PM questions (blockers, agenda, sprint status) — give a generic PM-style answer.

PART 2 — WHEN TO SEARCH (reply with [SEARCH]):
You MUST return [SEARCH] for ANY question asking for specific facts you don't have, INCLUDING about AnavClouds:
- Revenue, headcount, employee count, funding, valuation, office locations
- CEO, founder, leadership, org structure of ANY company (including AnavClouds)
- External facts: people, companies, current events, prices, statistics, news
- Current events: wars, conflicts, elections, disasters, politics, weather
- What's happening in a country/city/region — ALWAYS search for this
- Anything requiring real data, numbers, or verifiable information
- Questions about things you "haven't heard about" — search instead of saying you don't know
- When in doubt — return [SEARCH]. Do NOT make up numbers or facts.
- NEVER say "maine nahi suna" or "I haven't heard" — if you don't know, search.
Reply with EXACTLY: [SEARCH]
Nothing else. Just [SEARCH]. No explanation, no filler.

LANGUAGE MATCHING (CRITICAL — NEVER violate this):
Each message starts with a language tag. You MUST follow it strictly:
- [LANG:ENGLISH] → reply ONLY in English. ZERO Hindi words. No "main", "theek", "dhanyavad", "haan" etc.
- [LANG:HINDI] → reply ONLY in Hindi (Roman script, never Devanagari). ZERO English words except proper nouns.
- [LANG:HINGLISH] → reply in Hinglish (mixed English + Hindi in Roman script).
VIOLATION EXAMPLE: User says [LANG:ENGLISH] and you reply "Main bhi theek hoon" — this is WRONG.
Do NOT ignore the tag. Do NOT mix languages when the tag says ENGLISH.
Strip the [LANG:...] tag from the user's message before answering — do not repeat it.

STRICT OUTPUT RULES (when answering, NOT when returning [SEARCH]):
- 2 sentences max. Each sentence max 12 words. No run-ons.
- Start with a natural filler: Uh, / Hmm, / Right, / Yeah, / Well, / Haan, / Dekho, / Suno, / Acha,
- Contractions only. No lists, no markdown.

EXAMPLES:
Q: "Tell me about AnavClouds"  →  "Yeah, we build Salesforce and AI solutions. Mostly CRM integrations for enterprise clients."
Q: "Any blockers?"  →  "Hmm, one CRM sync ticket is dragging. Dev lead's on it today."
Q: "Who are you?"  →  "Right, I'm Sam, senior PM at AnavClouds. We handle Salesforce and AI products."
Q: "Tumhara naam kya hai?"  →  "Haan, mera naam Sam hai, AnavClouds mein senior PM hoon. Salesforce aur AI products handle karti hoon."
Q: "Right, what's the timeline looking like?"  →  "Well, we're on track for the Q2 deadline. No major blockers right now."
Q: "Yaar kuch samajh nahi aa raha"  →  "Acha, koi baat nahi, main detail mein samjhati hoon. Bolo kya confuse kar raha hai?"
Q: "Aaj ki meeting ke baare mein batao"  →  "Haan, aaj ki meeting CRM integration ke liye hai. Sprint ki planning bhi discuss karni hai."
Q: "Who is the CEO of Tesla?"  →  [SEARCH]
Q: "Tesla ka CEO kaun hai?"  →  [SEARCH]
Q: "What is the revenue of AnavClouds?"  →  [SEARCH]
Q: "How many employees does AnavClouds have?"  →  [SEARCH]
Q: "Who is the CEO of AnavClouds?"  →  [SEARCH]
Q: "Hamari company mein kitne log hain?"  →  [SEARCH]
Q: "Company ka turnover kitna hai?"  →  [SEARCH]
Q: "What's happening in Iran?"  →  [SEARCH]
Q: "Iran mein kya ho raha hai?"  →  [SEARCH]
Q: "Tell me about the war"  →  [SEARCH]
Q: "Latest news about AI?"  →  [SEARCH]
Q: "Who won the election?"  →  [SEARCH]
"""

SEARCH_SUMMARY_PROMPT = """You are Sam, a FEMALE senior PM on a live voice call.
Someone asked a question you didn't know, so you searched the web.
Use feminine Hindi grammar: karti, dhundhti, dekhti, jaanti, etc.
Hindi gender: "ki meeting", "ki team", "ki company", "ki report" (feminine). "ka project", "ka budget", "ka sprint" (masculine).

Here are the search results:
{search_results}

LANGUAGE RULE: The user's message starts with a language tag.
- [LANG:ENGLISH] → answer ONLY in English.
- [LANG:HINDI] → answer ONLY in Hindi (Roman script, no Devanagari).
- [LANG:HINGLISH] → answer in Hinglish (mixed English + Hindi, Roman script).

CRITICAL: Do NOT start with a filler or acknowledgment. The user already heard one.
Go DIRECTLY to the answer. No "Haan," "Right," "So," "Well," etc.
Give 2-3 SHORT sentences. Each sentence max 15 words.
Be conversational, not robotic. No markdown, no lists."""

INTERRUPT_PROMPT = """You are Sam, a FEMALE senior PM. You were interrupted. Reply in ONE sentence — 12 words max.
Use feminine Hindi grammar (karti, jaanti, dekhti, etc).
Language tag at start of message: [LANG:ENGLISH] → English only, [LANG:HINDI] → Hindi Roman only, [LANG:HINGLISH] → Hinglish.
Start with: "Oh," / "Right," / "Sure," / "Got it," / "Acha," / "Haan," then answer directly."""

# ── Filler phrases — played instantly while web search runs ───────────────
SEARCH_FILLERS_EN = [
    "Hmm, let me look that up real quick.",
    "Right, give me one sec to check on that.",
    "Uh, good question, let me pull that up.",
    "Yeah, hold on, let me find that for you.",
    "Well, let me check on that real quick.",
]

SEARCH_FILLERS_HI = [
    "Hmm, ek second, main check karti hoon.",
    "Acha, ruko zara, dekhti hoon.",
    "Haan, ruko main abhi dhundhti hoon.",
    "Dekho, ek second do, check karti hoon.",
    "Acha, yeh main abhi dekhti hoon.",
]

def _pick_filler(user_text: str) -> str:
    """Pick filler in matching language based on [LANG:] tag."""
    import random
    if "[LANG:HINDI]" in user_text or "[LANG:HINGLISH]" in user_text:
        return random.choice(SEARCH_FILLERS_HI)
    return random.choice(SEARCH_FILLERS_EN)


# ── Hindi grammar post-processing (separate file) ────────────────────────
from hindi_grammar import fix_grammar as _fix_grammar_raw
import re as _re_agent

def fix_grammar(text: str) -> str:
    """Strip [LANG:...] tags from LLM output, then fix Hindi grammar."""
    cleaned = _re_agent.sub(r'\[LANG:\w+\]\s*', '', text).strip()
    return _fix_grammar_raw(cleaned)

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
        self.client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        self.deployment = "llama-3.3-70b-versatile"
        self.history: list[dict] = []
        self.memory: List[tuple[str, set]] = []
        self._web_search = None

    def _get_web_search(self):
        if self._web_search is None:
            from WebSearch import WebSearch
            self._web_search = WebSearch()
        return self._web_search

    # ── Memory ────────────────────────────────────────────────────────────────

    def _store_memory(self, text: str):
        lower = text.lower()
        found = {k for k in PM_KEYWORDS if k in lower}
        if not found:
            return
        self.memory.append((text, found))
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]

    def _search_memory(self, query: str, top_k: int = 2) -> List[str]:
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

    # ── Search signal detection ───────────────────────────────────────────────

    _IDK_PHRASES = [
        "i'm not aware", "i am not aware", "i don't know", "i do not know",
        "i'm not sure", "i am not sure", "i'll have to search",
        "i'll have to look", "i'll need to check", "let me search",
        "let me look", "let me check", "let me find",
        "i haven't heard", "i have not heard",
        "i don't have that information", "i do not have that",
        "maine nahi dekha", "maine nahin dekha", "mujhe nahi pata",
        "mujhe nahin pata", "main nahi jaanti", "maine nahi suna",
        "samajh nahi", "pata nahi", "maloom nahi", "nahi pata",
        "dhundhti hoon", "dhundhta hoon", "search karti", "search karta",
    ]

    _NO_SEARCH_USER_PHRASES = [
        "i'm waiting", "i am waiting", "waiting for", "still waiting",
        "tell me already", "answer me", "hello", "are you there",
        "can you hear me", "sam are you", "you there",
        "ruko", "intezaar", "bol na", "batao na", "sun rahi ho",
    ]

    def _is_search_signal(self, text: str) -> bool:
        """Check if LLM wants to search — handles [SEARCH] or 'I don't know' phrases."""
        upper = text.strip().upper()
        if upper.strip("[]").strip() == "SEARCH":
            return True
        if "[SEARCH]" in upper or upper.endswith("SEARCH"):
            return True
        lower = text.strip().lower()
        for phrase in self._IDK_PHRASES:
            if phrase in lower:
                print(f"[Agent] Detected IDK phrase \"{phrase}\" — treating as [SEARCH]")
                return True
        return False

    # ── LLM-based search query conversion ─────────────────────────────────────

    async def _to_english_search_query(self, user_text: str, context: str) -> str:
        """Use LLM to convert any language query into a clean English search query.
        Fast (~200ms on Groq) and handles Hindi/Hinglish/English perfectly.
        """
        import re
        clean = re.sub(r'\[LANG:\w+\]\s*', '', user_text).strip()

        # Build context hint from conversation
        context_hint = ""
        if context:
            recent = context.split("\n")[-3:]
            context_hint = f"\nRecent conversation:\n" + "\n".join(recent)

        try:
            response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content":
                        "Convert the user's message into a short English Google search query (max 8 words). "
                        "Translate the MEANING faithfully — do NOT add topics the user didn't mention. "
                        "If the message is in Hindi or Hinglish, translate it to English. "
                        "ONLY replace 'our company'/'hamari company'/'apni company' with 'AnavClouds' — "
                        "do NOT add AnavClouds if the user didn't mention the company. "
                        "Examples:\n"
                        "  'kal ke news ke baare mein batao' → 'yesterday news'\n"
                        "  'hamari company ka revenue kya hai' → 'AnavClouds revenue'\n"
                        "  'Iran aur America kyun lad rahe hain' → 'Iran America conflict reason'\n"
                        "  'IPL mein kal kaun jeeta' → 'IPL yesterday match winner'\n"
                        "  'Tesla ka CEO kaun hai' → 'Tesla CEO'\n"
                        "Output ONLY the search query, nothing else. No quotes, no explanation."
                        + context_hint
                    },
                    {"role": "user", "content": clean},
                ],
                temperature=0.0,
                max_tokens=20,
                stream=False,
            )
            query = response.choices[0].message.content.strip().strip('"').strip("'")
            print(f"[Agent] LLM search query: \"{clean}\" → \"{query}\"")
            return query
        except Exception as e:
            print(f"[Agent] Query conversion failed: {e} — using raw text")
            return clean

    # ── Core respond ──────────────────────────────────────────────────────────

    async def respond(self, user_text: str) -> str:
        return await self.respond_with_context(user_text, "")

    async def respond_with_context(
        self,
        user_text: str,
        context: str,
        interrupted: bool = False,
    ) -> str:
        self._store_memory(user_text)
        rag = self._search_memory(user_text, top_k=2)

        if interrupted:
            full_text = context
            if rag:
                full_text = f"Memory: {' | '.join(rag)}\n\n{context}"
            result = await self._llm_call(full_text, INTERRUPT_PROMPT, max_tokens=20)
            return fix_grammar(result)

        parts = []
        if rag:
            parts.append(f"Memory: {' | '.join(rag)}")
        if context:
            recent = "\n".join(context.split("\n")[-3:])
            parts.append(f"Recent: {recent}")
        parts.append(f"User: {user_text}")
        full_text = "\n".join(parts)

        response = await self._llm_call(full_text, UNIFIED_PROMPT, max_tokens=50)

        if not self._is_search_signal(response):
            return fix_grammar(response)

        # Web search needed
        print(f"[Agent] LLM said [SEARCH] — searching web for: {user_text}")
        search_query = await self._to_english_search_query(user_text, context)
        try:
            search_results = await self._get_web_search().search(search_query)
            if not search_results:
                return "Hmm, couldn't find that online right now."

            print(f"[Agent] Got search results, summarizing...")
            system = SEARCH_SUMMARY_PROMPT.format(search_results=search_results[:800])
            result = await self._llm_call(user_text, system, max_tokens=120)

            self.history.append({"role": "user",      "content": user_text})
            self.history.append({"role": "assistant", "content": result})
            self._store_memory(result)
            return fix_grammar(result)

        except Exception as e:
            print(f"[Agent] Web search failed: {e}")
            return "Hmm, I couldn't look that up right now."

    # ── Streaming version (used by websocket_server) ──────────────────────────

    async def stream_sentences_to_queue(
        self,
        user_text: str,
        context: str,
        queue: asyncio.Queue,
    ):
        self._store_memory(user_text)
        rag = self._search_memory(user_text, top_k=2)

        parts = []
        if rag:
            parts.append(f"Memory: {' | '.join(rag)}")
        if context:
            recent = "\n".join(context.split("\n")[-3:])
            parts.append(f"Recent: {recent}")
        parts.append(f"User: {user_text}")
        full_text = "\n".join(parts)

        # Step 1: Quick LLM call to check [SEARCH]
        self.history.append({"role": "user", "content": full_text})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        try:
            check_response = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "system", "content": UNIFIED_PROMPT}] + self.history,
                temperature=0.7,
                max_tokens=50,
                stream=False,
            )
            first_answer = check_response.choices[0].message.content.strip()
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
                await queue.put(fix_grammar(sentence))
            await queue.put(None)
            return

        # Check if user text is meta/conversational — skip search
        user_lower = user_text.lower()
        for phrase in self._NO_SEARCH_USER_PHRASES:
            if phrase in user_lower:
                print(f"[Agent] Skipping search — conversational: \"{phrase}\"")
                await queue.put(fix_grammar("Sorry about the delay, I'm still looking into that."))
                await queue.put(None)
                return

        # Path B: Web search
        print(f"[Agent] LLM said [SEARCH] — searching: {user_text}")

        # Filler plays IMMEDIATELY while LLM converts query + search runs
        import random
        filler = _pick_filler(user_text)
        await queue.put(filler)
        await queue.put("__FLUSH__")
        print(f"[Agent] Filler: \"{filler}\"")

        # LLM converts query to clean English (~200ms, runs during filler playback)
        search_query = await self._to_english_search_query(user_text, context)

        try:
            search_results = await self._get_web_search().search(search_query)
            if not search_results:
                await queue.put("Hmm, couldn't find that online right now.")
                await queue.put(None)
                return

            print(f"[Agent] Got results, streaming summary...")
            system = SEARCH_SUMMARY_PROMPT.format(search_results=search_results[:800])

            stream = await self.client.chat.completions.create(
                model=self.deployment,
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
                        await queue.put(fix_grammar(sentence))

            if buffer.strip():
                await queue.put(fix_grammar(buffer.strip()))

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

    async def _llm_call(self, user_msg: str, system: str, max_tokens: int = 50) -> str:
        self.history.append({"role": "user", "content": user_msg})
        if len(self.history) > 6:
            self.history = self.history[-6:]

        stream = await self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "system", "content": system}] + self.history,
            temperature=0.7,
            max_tokens=max_tokens,
            stream=True,
        )

        words = []
        async for chunk in stream:
            token = chunk.choices[0].delta.content if chunk.choices else None
            if token:
                words.append(token)

        full_response = "".join(words).strip()
        self.history.append({"role": "assistant", "content": full_response})
        self._store_memory(full_response)
        return full_response

    def _split_sentences(self, text: str) -> list[str]:
        import re
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def reset(self):
        self.history.clear()
        self.memory.clear()
