# """
# Trigger.py
# Fast-path triggers (no LLM call) cover ~80% of cases:
#   - Direct address ("sam") → YES immediately
#   - Trivial fillers → NO immediately
#   - Incomplete endings → wait
#   - Memory recall keywords → YES immediately
#   - Question ending "?" → YES immediately
#   - 2+ PM keywords → YES immediately
#   - Recent follow-up (< 3s) → YES immediately
#   - Cooldown gate → NO
#   - Ambiguous phrases → LLM decision (~200-700ms)
# """

# import time
# import os
# from openai import AsyncOpenAI

# COOLDOWN_SECONDS = 1.5

# TRIGGER_PROMPT = """You are Sam, a senior Project Manager in a live meeting.
# Your job is to decide whether YOU should speak next.

# Context:
# {context}

# Relevant past memory:
# {memory}

# Latest message from {speaker}: "{text}"

# Say YES if:
# - The message is directed at you explicitly or implicitly
# - It is a question expecting your input
# - It refers to something discussed earlier
# - It feels like a follow-up to something you said
# - The speaker seems to be waiting for your response
# - Someone greets the group — greet back

# Say NO if:
# - The message is clearly directed to someone else by name
# - Two other people are talking to each other
# - It is just filler, acknowledgment, or side chatter
# - The speaker is clearly mid-sentence (ends with "and", "so", "but", "the")

# Reply ONLY with YES or NO.
# """

# PM_KEYWORDS = [
#     "deadline", "deliver", "blocker", "issue", "plan", "decide",
#     "approved", "timeline", "task", "owner", "risk", "budget",
#     "scope", "stakeholder", "milestone", "sprint", "feature",
#     "requirement", "sign-off", "contract", "report", "project",
#     "team", "priority", "update", "review", "status", "delay",
#     "launch", "release", "client", "dependency", "estimate",
# ]

# FILLERS = {
#     "okay", "ok", "sure", "thanks", "thank you", "yep", "nope",
#     "alright", "hmm", "uh huh", "got it", "bye", "yeah", "yes",
#     "no", "cool", "nice", "great", "perfect", "sounds good",
#     "i see", "right", "okay okay", "ok ok",
# }

# INCOMPLETE_ENDINGS = {
#     "and", "so", "then", "but", "the", "a", "an", "or", "if", "when"
# }

# RECALL_KEYWORDS = [
#     "before", "earlier", "told you", "mentioned",
#     "remember", "what did i say", "recall", "last time",
#     "previously", "you said",
# ]


# class TriggerDetector:
#     def __init__(self):
#         self._last_response_at: float = 0.0
#         self._client = AsyncOpenAI(
#             api_key=os.environ["GROQ_API_KEY"],
#             base_url="https://api.groq.com/openai/v1",
#         )

#     async def should_respond(
#         self,
#         text: str,
#         speaker: str = "Unknown",
#         context: str = "",
#         memory: list[str] | None = None,
#     ) -> bool:
#         now   = time.monotonic()
#         lower = text.lower().strip()

#         # Direct address — always YES, instant
#         if "sam" in lower:
#             self._last_response_at = 0
#             print("  ⚡ Direct address — YES")
#             return True

#         # Trivial fillers — always NO, instant
#         if lower in FILLERS:
#             return False

#         # Incomplete sentence — wait
#         words     = lower.split()
#         last_word = words[-1] if words else ""
#         if last_word in INCOMPLETE_ENDINGS:
#             print("  ⏸ Incomplete — waiting")
#             return False

#         # Memory recall — always YES, instant
#         if any(k in lower for k in RECALL_KEYWORDS):
#             print("  🧠 Recall detected — YES")
#             return True

#         # Question ending — almost always directed at Sam in 1:1 context
#         if lower.endswith("?"):
#             print("  ❓ Question — YES")
#             return True

#         # 2+ PM keywords — relevant to Sam's domain
#         pm_hits = {k for k in PM_KEYWORDS if k in lower}
#         if len(pm_hits) >= 2:
#             print(f"  🏷️  PM keywords ({pm_hits}) — YES")
#             return True

#         # Recent follow-up boost
#         if now - self._last_response_at < 3:
#             print("  🔁 Follow-up boost — YES")
#             return True

#         # Cooldown gate
#         if now - self._last_response_at < COOLDOWN_SECONDS:
#             return False

#         # LLM decision — only hits here for ambiguous phrases
#         memory_hint = "\n".join(memory[-5:]) if memory else "None"
#         return await self._groq_decide(text, speaker, context, memory_hint)

#     async def _groq_decide(
#         self, text: str, speaker: str, context: str, memory: str
#     ) -> bool:
#         try:
#             response = await self._client.chat.completions.create(
#                 model="llama-3.1-8b-instant",
#                 messages=[{
#                     "role": "user",
#                     "content": TRIGGER_PROMPT.format(
#                         context=context or "No prior context",
#                         speaker=speaker,
#                         text=text,
#                         memory=memory,
#                     )
#                 }],
#                 temperature=0,
#                 max_tokens=3,
#             )
#             decision = response.choices[0].message.content.strip().upper()
#             return "YES" in decision

"""
Trigger.py — OPTIMIZED
Changes:
  1. Broader fast-path patterns ("about you", "tell me", etc.) → instant YES
  2. Groq LLM fallback capped at 500ms → default YES on timeout
"""

import time
import os
import asyncio
from openai import AsyncOpenAI

COOLDOWN_SECONDS = 1.5

TRIGGER_PROMPT = """You are Sam, a senior Project Manager in a live meeting.
Your job is to decide whether YOU should speak next.

Context:
{context}

Relevant past memory:
{memory}

Latest message from {speaker}: "{text}"

Say YES if:
- The message is directed at you explicitly or implicitly
- It is a question expecting your input
- It refers to something discussed earlier
- It feels like a follow-up to something you said
- The speaker seems to be waiting for your response
- Someone greets the group — greet back

Say NO if:
- The message is clearly directed to someone else by name
- Two other people are talking to each other
- It is just filler, acknowledgment, or side chatter
- The speaker is clearly mid-sentence (ends with "and", "so", "but", "the")

Reply ONLY with YES or NO.
"""

PM_KEYWORDS = [
    "deadline", "deliver", "blocker", "issue", "plan", "decide",
    "approved", "timeline", "task", "owner", "risk", "budget",
    "scope", "stakeholder", "milestone", "sprint", "feature",
    "requirement", "sign-off", "contract", "report", "project",
    "team", "priority", "update", "review", "status", "delay",
    "launch", "release", "client", "dependency", "estimate",
]

FILLERS = {
    "okay", "ok", "sure", "thanks", "thank you", "yep", "nope",
    "alright", "hmm", "uh huh", "got it", "bye", "yeah", "yes",
    "no", "cool", "nice", "great", "perfect", "sounds good",
    "i see", "right", "okay okay", "ok ok",
    # Extended fillers
    "interesting", "i see", "noted", "understood", "makes sense",
    "fair enough", "true", "exactly", "absolutely", "definitely",
    "of course", "certainly", "indeed", "wow", "oh", "ah",
    "mhm", "mm", "uh", "um", "hm", "okay sam", "ok sam",
    "stop", "stop sam", "wait", "hold on", "one sec", "one second",
}

INCOMPLETE_ENDINGS = {
    "and", "so", "then", "but", "the", "a", "an", "or", "if", "when"
}

RECALL_KEYWORDS = [
    "before", "earlier", "told you", "mentioned",
    "remember", "what did i say", "recall", "last time",
    "previously", "you said",
]

# NEW: broader patterns that clearly need Sam to respond
DIRECT_ADDRESS_PATTERNS = [
    "sam", "about you", "about yourself", "tell me",
    "what do you", "can you", "could you", "would you",
    "your opinion", "your thoughts", "what's your",
    "introduce yourself", "who are you",
]


class TriggerDetector:
    def __init__(self):
        self._last_response_at: float = 0.0
        self._client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

    async def should_respond(
        self,
        text: str,
        speaker: str = "Unknown",
        context: str = "",
        memory: list[str] | None = None,
    ) -> bool:
        now   = time.monotonic()
        lower = text.lower().strip()

        # Direct address patterns — instant YES
        if any(p in lower for p in DIRECT_ADDRESS_PATTERNS):
            self._last_response_at = 0
            print("  ⚡ Direct address — YES")
            return True

        # Trivial fillers — instant NO
        if lower in FILLERS:
            return False

        # Incomplete sentence — wait
        words     = lower.split()
        last_word = words[-1] if words else ""
        if last_word in INCOMPLETE_ENDINGS:
            print("  ⏸ Incomplete — waiting")
            return False

        # Memory recall — instant YES
        if any(k in lower for k in RECALL_KEYWORDS):
            print("  🧠 Recall detected — YES")
            return True

        # Question ending — YES
        if lower.endswith("?"):
            print("  ❓ Question — YES")
            return True

        # 2+ PM keywords — YES
        pm_hits = {k for k in PM_KEYWORDS if k in lower}
        if len(pm_hits) >= 2:
            print(f"  🏷️  PM keywords ({pm_hits}) — YES")
            return True

        # Recent follow-up boost
        if now - self._last_response_at < 3:
            print("  🔁 Follow-up boost — YES")
            return True

        # Cooldown gate
        if now - self._last_response_at < COOLDOWN_SECONDS:
            return False

        # LLM decision — CAPPED at 500ms, default YES on timeout
        memory_hint = "\n".join(memory[-5:]) if memory else "None"
        return await self._groq_decide(text, speaker, context, memory_hint)

    async def _groq_decide(
        self, text: str, speaker: str, context: str, memory: str
    ) -> bool:
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{
                        "role": "user",
                        "content": TRIGGER_PROMPT.format(
                            context=context or "No prior context",
                            speaker=speaker,
                            text=text,
                            memory=memory,
                        )
                    }],
                    temperature=0,
                    max_tokens=3,
                ),
                timeout=0.5,  # 500ms hard cap
            )
            decision = response.choices[0].message.content.strip().upper()
            result = "YES" in decision
            print(f"  🤖 LLM trigger: {'YES' if result else 'NO'}")
            return result
        except asyncio.TimeoutError:
            print("  ⏱️  LLM trigger timeout — defaulting YES")
            return True
        except Exception as e:
            print(f"[Trigger] Error: {e}")
            return text.strip().endswith("?")

    def mark_responded(self):
        self._last_response_at = time.monotonic()
