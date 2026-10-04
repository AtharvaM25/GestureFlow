"""
Measure WebSocket recognition latency against a running server, with real landmarks.

    python -m backend.bench_ws --url http://localhost:8000 --frames 600

Registers a throwaway user, opens a session and streams frames from data/landmarks.csv.
Reports the server's own per-frame time and the client-measured round trip (p50/p95/p99).
"""

import argparse
import asyncio
import json
import secrets
import statistics
import time

import httpx
import numpy as np
import pandas as pd
import websockets

from gestureflow.paths import LANDMARKS_CSV

COLS = [f"{a}{i}" for i in range(21) for a in "xy"]


def pct(values, p):
    return float(np.percentile(values, p))


async def run(url: str, frames: int):
    email = f"bench-{secrets.token_hex(4)}@example.com"
    password = secrets.token_urlsafe(12)
    async with httpx.AsyncClient(base_url=url) as http:
        (await http.post("/api/v1/auth/register",
                         json={"email": email, "password": password})).raise_for_status()
        tokens = (await http.post("/api/v1/auth/login",
                                  json={"email": email, "password": password})).json()
        auth = {"Authorization": f"Bearer {tokens['access_token']}"}
        session = (await http.post("/api/v1/sessions", headers=auth, json={})).json()

    df = pd.read_csv(LANDMARKS_CSV).sample(frames, replace=True, random_state=0)
    pts = df[COLS].to_numpy(np.float32).reshape(-1, 21, 2).tolist()

    server_ms, rtt_ms = [], []
    ws_url = url.replace("http", "ws", 1) + "/api/v1/ws/recognition"
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps({"type": "auth", "token": tokens["access_token"],
                                  "session_id": session["id"]}))
        json.loads(await ws.recv())
        for i, landmarks in enumerate(pts):
            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "frame", "landmarks": landmarks, "client_ts": i}))
            msg = json.loads(await ws.recv())
            rtt_ms.append((time.perf_counter() - t0) * 1000)
            if i >= 20:                     # skip warm-up frames
                server_ms.append(msg["latency_ms"])
            else:
                rtt_ms.pop()

    async with httpx.AsyncClient(base_url=url) as http:
        await http.post(f"/api/v1/sessions/{session['id']}/end", headers=auth)

    print(f"frames measured: {len(server_ms)} (after 20 warm-up)")
    for name, v in (("server per-frame", server_ms), ("round trip", rtt_ms)):
        print(f"  {name:<17} p50 {pct(v, 50):6.2f} ms   p95 {pct(v, 95):6.2f} ms   "
              f"p99 {pct(v, 99):6.2f} ms   mean {statistics.mean(v):6.2f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()
    asyncio.run(run(args.url, args.frames))


if __name__ == "__main__":
    main()
