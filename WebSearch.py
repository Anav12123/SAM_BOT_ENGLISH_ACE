"""
WebSearch.py
Uses SerpAPI Google search — returns answer boxes, knowledge graphs, or organic results.
Rotates between 2 API keys each call to avoid rate limits.
Query conversion (Hinglish→English) is handled by Agent.py via LLM.

Set env vars:
  SERPAPI_KEY_1=your_first_key
  SERPAPI_KEY_2=your_second_key
"""

import os
import re
import httpx
from typing import Optional


class WebSearch:
    def __init__(self):
        self._keys = []
        k1 = os.environ.get("SERPAPI_KEY_1", "").strip().strip('"').strip("'")
        k2 = os.environ.get("SERPAPI_KEY_2", "").strip().strip('"').strip("'")
        if k1:
            self._keys.append(k1)
        if k2:
            self._keys.append(k2)

        if not self._keys:
            print("[WebSearch] ⚠️  No SERPAPI keys set — web search disabled")
        else:
            print(f"[WebSearch] {len(self._keys)} SerpAPI key(s) loaded (key1: {k1[:8]}...)")

        self._key_index = 0
        self._client = httpx.AsyncClient(timeout=20.0)

    def _next_key(self) -> str:
        """Rotate between available keys."""
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def _trim_query(self, query: str, max_words: int = 25) -> str:
        """Strip language tags, filler prefixes, and cap at max_words.
        Query is already in English (LLM converted it in Agent.py).
        """
        clean = query.strip()
        clean = re.sub(r'\[LANG:\w+\]\s*', '', clean)
        for prefix in ["sam,", "sam ", "hey sam,", "hey sam ",
                        "can you tell me", "could you tell me",
                        "please tell me", "do you know",
                        "i want to know", "tell me"]:
            if clean.lower().startswith(prefix):
                clean = clean[len(prefix):].strip().lstrip(",. ")

        words = clean.split()
        if len(words) > max_words:
            words = words[:max_words]
        return " ".join(words)

    async def search(self, query: str) -> Optional[str]:
        """
        Search via SerpAPI Google search.
        Returns answer box / knowledge graph / AI overview / organic results.
        """
        if not self._keys:
            return None

        trimmed = self._trim_query(query)
        api_key = self._next_key()

        print(f"[WebSearch] SerpAPI query: \"{trimmed}\" (key #{self._key_index})")

        try:
            response = await self._client.get(
                "https://serpapi.com/search.json",
                params={
                    "engine":  "google",
                    "q":       trimmed,
                    "api_key": api_key,
                    "num":     3,
                },
            )
            if response.status_code != 200:
                print(f"[WebSearch] SerpAPI HTTP {response.status_code}: {response.text[:200]}")
                return None
            data = response.json()

            # 1. Answer box (best — direct answer)
            answer_box = data.get("answer_box", {})
            answer = answer_box.get("answer", "") or answer_box.get("snippet", "")
            if answer:
                print(f"[WebSearch] Got answer box ({len(answer)} chars)")
                return answer[:800]

            # 2. Knowledge graph description
            kg = data.get("knowledge_graph", {})
            kg_desc = kg.get("description", "")
            if kg_desc:
                title = kg.get("title", "")
                result = f"{title}: {kg_desc}" if title else kg_desc
                print(f"[WebSearch] Got knowledge graph ({len(result)} chars)")
                return result[:800]

            # 3. AI overview / featured snippet
            ai_overview = data.get("ai_overview", {})
            if ai_overview:
                blocks = ai_overview.get("text_blocks", [])
                parts = []
                for block in blocks:
                    snippet = block.get("snippet", "")
                    if snippet:
                        parts.append(snippet)
                if parts:
                    combined = " ".join(parts)[:800]
                    print(f"[WebSearch] Got AI overview ({len(combined)} chars)")
                    return combined

            # 4. Top organic results
            organic = data.get("organic_results", [])
            parts = []
            for r in organic[:3]:
                snippet = r.get("snippet", "")
                if snippet:
                    parts.append(snippet)
            if parts:
                combined = " ".join(parts)[:800]
                print(f"[WebSearch] Got organic results ({len(combined)} chars)")
                return combined

            return None

        except httpx.TimeoutException:
            print(f"[WebSearch] SerpAPI TIMEOUT for: {trimmed}")
            return None
        except Exception as e:
            print(f"[WebSearch] SerpAPI failed: {type(e).__name__}: {e}")
            return None

    async def close(self):
        await self._client.aclose()
