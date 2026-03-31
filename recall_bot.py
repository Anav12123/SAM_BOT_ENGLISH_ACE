"""
recall_bot.py — Recall.ai bot with Deepgram Nova-3 (English)
"""

import os
import httpx

RECALL_REGION   = os.environ.get("RECALLAI_REGION", "ap-northeast-1")
RECALL_API_BASE = f"https://{RECALL_REGION}.recall.ai/api/v1"

SILENT_MP3_B64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjI5LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAADkADMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzM//////////////////////////////////////////////////////////////////8AAAAATGF2YzU4LjU0AAAAAAAAAAAAAAAAJAAAAAAAAAAAkFCGaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


class RecallBot:
    def __init__(self):
        self.api_key = os.environ["RECALLAI_API_KEY"]
        self.bot_id: str | None = None
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type":  "application/json",
            "accept":        "application/json",
        }

    async def join(self, meeting_url: str, websocket_url: str) -> str:
        platform = "Google Meet" if "meet.google" in meeting_url else \
                   "Teams" if "teams.microsoft" in meeting_url else \
                   "Zoom" if "zoom.us" in meeting_url else "Unknown"
        print(f"[Recall.ai] Joining {platform} meeting...")

        payload = {
            "meeting_url": meeting_url,
            "bot_name":    "Sam",
            "recording_config": {
                "transcript": {
                    "provider": {
                        "deepgram_streaming": {
                            "language":    "en",
                            "model":       "nova-3",
                            "endpointing": 100,
                            "keyterm": [
                                "AnavClouds",
                                "AnavClouds Software Solutions",
                                "Salesforce",
                                "Sam"
                            ]
                        }
                    }
                },
                "audio_separate_raw": {},
                "realtime_endpoints": [{
                    "type":   "websocket",
                    "url":    websocket_url,
                    "events": [
                        "transcript.data",
                        "transcript.partial_data",
                        "participant_events.speech_on",
                        "participant_events.speech_off",
                        "participant_events.join",
                        "participant_events.leave",
                        "audio_separate_raw.data",
                    ]
                }]
            },
            "automatic_audio_output": {
                "in_call_recording": {
                    "data": {"kind": "mp3", "b64_data": SILENT_MP3_B64}
                }
            }
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{RECALL_API_BASE}/bot/",
                headers=self.headers,
                json=payload,
            )
            if resp.status_code != 201:
                print(f"[Recall.ai] Error: {resp.text}")
            resp.raise_for_status()
            data = resp.json()

        self.bot_id = data["id"]
        print(f"[Recall.ai] Bot joined! ID: {self.bot_id}")
        return self.bot_id

    async def leave(self):
        if not self.bot_id:
            return
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{RECALL_API_BASE}/bot/{self.bot_id}/leave_call/",
                headers=self.headers,
            )
        print("[Recall.ai] Bot left the meeting.")
        self.bot_id = None

    async def get_status(self) -> dict:
        if not self.bot_id:
            return {}
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{RECALL_API_BASE}/bot/{self.bot_id}/",
                headers=self.headers,
            )
            r.raise_for_status()
            return r.json()
