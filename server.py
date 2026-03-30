"""
server.py — Railway entry point
Starts the WebSocket server and exposes HTTP endpoints to control the bot.

POST /start  {"meeting_url": "..."}  → Sam joins the meeting
POST /stop                           → Sam leaves the meeting
GET  /health                         → health check
"""

import asyncio
import os
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

from websocket_server import WebSocketServer
from recall_bot import RecallBot

PORT = int(os.environ.get("PORT", 8000))

active_bot    = None
active_server = None


async def handle_start(request: web.Request) -> web.Response:
    global active_bot, active_server
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    meeting_url = data.get("meeting_url")
    if not meeting_url:
        return web.json_response({"error": "meeting_url required"}, status=400)

    # Railway sets RAILWAY_PUBLIC_DOMAIN automatically
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    if not domain:
        return web.json_response({"error": "RAILWAY_PUBLIC_DOMAIN not set"}, status=500)

    ws_url = f"wss://{domain}/ws"
    print(f"[Server] Sending Sam to {meeting_url}")
    print(f"[Server] WebSocket URL: {ws_url}")

    bot    = RecallBot()
    bot_id = await bot.join(meeting_url, ws_url)

    if active_server:
        active_server.speaker.bot_id = bot_id

    active_bot = bot
    return web.json_response({"status": "joined", "bot_id": bot_id})


async def handle_stop(request: web.Request) -> web.Response:
    global active_bot
    if active_bot:
        await active_bot.leave()
        active_bot = None
        return web.json_response({"status": "left"})
    return web.json_response({"status": "no active bot"})


async def handle_status(request: web.Request) -> web.Response:
    return web.json_response({
        "status":    "ok",
        "bot_active": active_bot is not None,
        "bot_id":    active_bot.bot_id if active_bot else None,
    })


async def main():
    global active_server

    server = WebSocketServer(port=PORT, bot_id=None)
    active_server = server

    # Add HTTP control routes to the same aiohttp app
    server.app.router.add_post("/start",  handle_start)
    server.app.router.add_post("/stop",   handle_stop)
    server.app.router.add_get("/health",  handle_status)

    await server.start()

    print(f"[Server] Running on port {PORT}")
    print(f"[Server] POST /start {{\"meeting_url\": \"...\"}} to deploy Sam")
    print(f"[Server] POST /stop to remove Sam")

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    asyncio.run(main())