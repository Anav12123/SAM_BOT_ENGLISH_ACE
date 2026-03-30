# # """
# # src/transcriber.py
# # Streams microphone audio to Deepgram Nova-2 and fires a callback
# # on every interim + final transcript.
# # """

# # import asyncio
# # import json
# # import os
# # import pyaudio
# # import websockets
# # from typing import Callable, Awaitable

# # DEEPGRAM_URL = (
# #     "wss://api.deepgram.com/v1/listen"
# #     "?model=nova-2"
# #     "&language=en-US"
# #     "&encoding=linear16"
# #     "&sample_rate=16000"
# #     "&channels=1"
# #     "&interim_results=true"
# #     "&endpointing=500"       # ms of silence → force a final result
# #     "&diarize=false"
# # )

# # CHUNK      = 8192   # audio frames per read
# # RATE       = 16000
# # FORMAT     = pyaudio.paInt16
# # CHANNELS   = 1


# # class DeepgramTranscriber:
# #     """
# #     Opens the microphone and forwards audio to Deepgram over a WebSocket.
# #     Calls `callback(text, is_final)` for every transcript event.
# #     """

# #     def __init__(self):
# #         self.api_key = os.environ["DEEPGRAM_API_KEY"]

# #     async def stream(
# #         self,
# #         callback: Callable[[str, bool], Awaitable[None]],
# #     ):
# #         headers = {"Authorization": f"Token {self.api_key}"}

# #         async with websockets.connect(DEEPGRAM_URL, extra_headers=headers) as ws:
# #             print("[Deepgram] Connected — listening...")

# #             # Microphone reader runs in a thread to avoid blocking the event loop
# #             audio_q: asyncio.Queue[bytes] = asyncio.Queue()
# #             loop = asyncio.get_event_loop()

# #             def mic_reader():
# #                 pa = pyaudio.PyAudio()
# #                 stream = pa.open(
# #                     format=FORMAT,
# #                     channels=CHANNELS,
# #                     rate=RATE,
# #                     input=True,
# #                     frames_per_buffer=CHUNK,
# #                 )
# #                 try:
# #                     while True:
# #                         data = stream.read(CHUNK, exception_on_overflow=False)
# #                         loop.call_soon_threadsafe(audio_q.put_nowait, data)
# #                 finally:
# #                     stream.stop_stream()
# #                     stream.close()
# #                     pa.terminate()

# #             # Start mic in background thread
# #             import threading
# #             t = threading.Thread(target=mic_reader, daemon=True)
# #             t.start()

# #             async def sender():
# #                 while True:
# #                     chunk = await audio_q.get()
# #                     await ws.send(chunk)

# #             async def receiver():
# #                 async for message in ws:
# #                     data = json.loads(message)
# #                     if data.get("type") != "Results":
# #                         continue
# #                     alts = data.get("channel", {}).get("alternatives", [])
# #                     if not alts:
# #                         continue
# #                     transcript = alts[0].get("transcript", "").strip()
# #                     if not transcript:
# #                         continue
# #                     is_final = data.get("is_final", False)
# #                     await callback(transcript, is_final)

# #             await asyncio.gather(sender(), receiver())

# """
# src/transcriber.py
# Streams microphone audio to Deepgram Nova-2 and fires a callback
# on every interim + final transcript.
# """

# import asyncio
# import json
# import os
# import pyaudio
# import websockets
# from typing import Callable, Awaitable

# DEEPGRAM_URL = (
#     "wss://api.deepgram.com/v1/listen"
#     "?model=nova-2"
#     "&language=en-US"
#     "&encoding=linear16"
#     "&sample_rate=16000"
#     "&channels=1"
#     "&interim_results=true"
#     "&endpointing=500"       # ms of silence → force a final result
#     "&diarize=false"
# )

# CHUNK      = 8192   # audio frames per read
# RATE       = 16000
# FORMAT     = pyaudio.paInt16
# CHANNELS   = 1


# class DeepgramTranscriber:
#     """
#     Opens the microphone and forwards audio to Deepgram over a WebSocket.
#     Calls `callback(text, is_final)` for every transcript event.
#     """

#     def __init__(self):
#         self.api_key = os.environ["DEEPGRAM_API_KEY"]

#     async def stream(
#         self,
#         callback: Callable[[str, bool], Awaitable[None]],
#     ):
#         headers = {"Authorization": f"Token {self.api_key}"}

#         async with websockets.connect(DEEPGRAM_URL, extra_headers=headers) as ws:
#             print("[Deepgram] Connected — listening...")

#             # Microphone reader runs in a thread to avoid blocking the event loop
#             audio_q: asyncio.Queue[bytes] = asyncio.Queue()
#             loop = asyncio.get_event_loop()

#             def mic_reader():
#                 pa = pyaudio.PyAudio()
#                 stream = pa.open(
#                     format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     frames_per_buffer=CHUNK,
#                 )
#                 try:
#                     while True:
#                         data = stream.read(CHUNK, exception_on_overflow=False)
#                         loop.call_soon_threadsafe(audio_q.put_nowait, data)
#                 finally:
#                     stream.stop_stream()
#                     stream.close()
#                     pa.terminate()

#             # Start mic in background thread
#             import threading
#             t = threading.Thread(target=mic_reader, daemon=True)
#             t.start()

#             async def sender():
#                 while True:
#                     chunk = await audio_q.get()
#                     await ws.send(chunk)

#             async def receiver():
#                 async for message in ws:
#                     data = json.loads(message)
#                     if data.get("type") != "Results":
#                         continue
#                     alts = data.get("channel", {}).get("alternatives", [])
#                     if not alts:
#                         continue
#                     transcript = alts[0].get("transcript", "").strip()
#                     if not transcript:
#                         continue
#                     is_final = data.get("is_final", False)
#                     await callback(transcript, is_final)

#             await asyncio.gather(sender(), receiver())

# """
# src/transcriber.py
# Streams microphone audio to Deepgram Nova-2 and fires a callback
# on every interim + final transcript.
# """

# import asyncio
# import json
# import os
# import pyaudio
# import websockets
# from typing import Callable, Awaitable

# DEEPGRAM_URL = (
#     "wss://api.deepgram.com/v1/listen"
#     "?model=nova-2"
#     "&language=en-US"
#     "&encoding=linear16"
#     "&sample_rate=16000"
#     "&channels=1"
#     "&interim_results=true"
#     "&endpointing=500"       # ms of silence → force a final result
#     "&diarize=false"
# )

# CHUNK      = 8192   # audio frames per read
# RATE       = 16000
# FORMAT     = pyaudio.paInt16
# CHANNELS   = 1


# class DeepgramTranscriber:
#     """
#     Opens the microphone and forwards audio to Deepgram over a WebSocket.
#     Calls `callback(text, is_final)` for every transcript event.
#     """

#     def __init__(self):
#         self.api_key = os.environ["DEEPGRAM_API_KEY"]

#     async def stream(
#         self,
#         callback: Callable[[str, bool], Awaitable[None]],
#     ):
#         headers = {"Authorization": f"Token {self.api_key}"}

#         async with websockets.connect(DEEPGRAM_URL, extra_headers=headers) as ws:
#             print("[Deepgram] Connected — listening...")

#             # Microphone reader runs in a thread to avoid blocking the event loop
#             audio_q: asyncio.Queue[bytes] = asyncio.Queue()
#             loop = asyncio.get_event_loop()

#             def mic_reader():
#                 pa = pyaudio.PyAudio()
#                 stream = pa.open(
#                     format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     frames_per_buffer=CHUNK,
#                 )
#                 try:
#                     while True:
#                         data = stream.read(CHUNK, exception_on_overflow=False)
#                         loop.call_soon_threadsafe(audio_q.put_nowait, data)
#                 finally:
#                     stream.stop_stream()
#                     stream.close()
#                     pa.terminate()

#             # Start mic in background thread
#             import threading
#             t = threading.Thread(target=mic_reader, daemon=True)
#             t.start()

#             async def sender():
#                 while True:
#                     chunk = await audio_q.get()
#                     await ws.send(chunk)

#             async def receiver():
#                 async for message in ws:
#                     data = json.loads(message)
#                     if data.get("type") != "Results":
#                         continue
#                     alts = data.get("channel", {}).get("alternatives", [])
#                     if not alts:
#                         continue
#                     transcript = alts[0].get("transcript", "").strip()
#                     if not transcript:
#                         continue
#                     is_final = data.get("is_final", False)
#                     await callback(transcript, is_final)

#             await asyncio.gather(sender(), receiver())

"""
src/transcriber.py
Streams microphone audio to Deepgram Nova-2 and fires a callback
on every interim + final transcript.
"""

import asyncio
import json
import os
import pyaudio
import websockets
from typing import Callable, Awaitable

DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"
    "&language=multi"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&interim_results=true"
    "&endpointing=500"       # ms of silence → force a final result
    "&diarize=false"
)

CHUNK      = 8192   # audio frames per read
RATE       = 16000
FORMAT     = pyaudio.paInt16
CHANNELS   = 1


class DeepgramTranscriber:
    """
    Opens the microphone and forwards audio to Deepgram over a WebSocket.
    Calls `callback(text, is_final)` for every transcript event.
    """

    def __init__(self):
        self.api_key = os.environ["DEEPGRAM_API_KEY"]

    async def stream(
        self,
        callback: Callable[[str, bool], Awaitable[None]],
    ):
        headers = {"Authorization": f"Token {self.api_key}"}

        async with websockets.connect(DEEPGRAM_URL, extra_headers=headers) as ws:
            print("[Deepgram] Connected — listening...")

            # Microphone reader runs in a thread to avoid blocking the event loop
            audio_q: asyncio.Queue[bytes] = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def mic_reader():
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                )
                try:
                    while True:
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        loop.call_soon_threadsafe(audio_q.put_nowait, data)
                finally:
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()

            # Start mic in background thread
            import threading
            t = threading.Thread(target=mic_reader, daemon=True)
            t.start()

            async def sender():
                while True:
                    chunk = await audio_q.get()
                    await ws.send(chunk)

            async def receiver():
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") != "Results":
                        continue
                    alts = data.get("channel", {}).get("alternatives", [])
                    if not alts:
                        continue
                    transcript = alts[0].get("transcript", "").strip()
                    if not transcript:
                        continue
                    is_final = data.get("is_final", False)
                    await callback(transcript, is_final)

            await asyncio.gather(sender(), receiver())