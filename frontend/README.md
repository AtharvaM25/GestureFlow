# GestureFlow frontend

Next.js 16 (App Router), TypeScript, Tailwind CSS 4 and shadcn/ui.

```bash
pnpm install        # also copies MediaPipe's runtime and model into public/mediapipe
pnpm dev            # http://localhost:3000, expects the API at NEXT_PUBLIC_API_URL
```

`NEXT_PUBLIC_API_URL` defaults to `http://localhost:8000`. It is inlined at build time, so
set it before `pnpm build`.

| Page | What it does |
| --- | --- |
| `/` | Landing page |
| `/login`, `/register` | Auth against `/api/v1/auth` |
| `/dashboard` | Your sessions and server status |
| `/recognize` | Webcam → MediaPipe in the browser → landmarks over WebSocket → live prediction |
| `/gestures` | The letters the model knows, with per-letter F1 on an unseen session |
| `/sessions/[id]` | Letters committed in one session |

Hand tracking runs in the browser. `src/lib/hand.ts` loads MediaPipe from `/mediapipe`,
which is served by this app, not a CDN, and converts landmarks to pixel coordinates of the
unmirrored frame before sending them. That is the space the model was trained in.

Checks: `pnpm lint`, `pnpm typecheck`, `pnpm build`.
