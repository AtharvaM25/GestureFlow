# Deploying GestureFlow

Four free services, connected in this order:

| Piece | Service | What you get |
| --- | --- | --- |
| Database | **Neon** (PostgreSQL) | a `postgresql://…` connection string |
| Sentence LLM | **Groq** | an API key |
| API + model | **Hugging Face Spaces** (free Gradio Space) | `https://<you>-gestureflow-api.hf.space` |
| Web app | **Vercel** | `https://<project>.vercel.app` |

Free tiers change; check each one's limits when you sign up. The API Space goes to sleep
when unused, and the first request after that takes up to a minute (the web app says so).

**Secrets go only into each service's settings page, never into GitHub.** `.env` is ignored
by git for that reason.

---

## 0. Before you start

Merge everything into `main` on GitHub and make sure the **CI** workflow is green.

## 1. Database — Neon

1. Sign up at neon.tech (GitHub login works) and create a project. Pick the region closest
   to your users.
2. On the project dashboard, copy the **connection string**. It looks like
   `postgresql://user:password@ep-…neon.tech/neondb?sslmode=require`.
   The API accepts it exactly as Neon gives it.
3. Keep it somewhere safe for step 3. Anyone with this string can read the whole database.

The tables are created automatically: the API runs `alembic upgrade head` every time it starts.

## 2. Sentence LLM — Groq

1. Sign up at console.groq.com and create an **API key**. Copy it; it starts with `gsk_`.
2. The default model is `llama-3.1-8b-instant` (small and fast). To use another, set
   `GESTUREFLOW_GROQ_MODEL` in step 3 to any model on Groq's model list.

## 3. API — Hugging Face Space

Docker Spaces need a paid plan, so the API runs in a free **Gradio** Space. Gradio is only
used for a small status page; the Space runs the same FastAPI app as everywhere else
(`deploy/huggingface/app.py`).

1. Sign up at huggingface.co → **New Space** (Manual setup):
   - Space name: `gestureflow-api`
   - SDK: **Gradio**, template **Blank**
   - Hardware: the free CPU tier (CPU basic)
   - Visibility: **Public** (a private Space's URL needs a token, which the browser won't have)
   - Create Space. It shows an empty Space; GitHub fills it in step 4.
2. In the Space → **Settings → Variables and secrets**, add:

   | Name | Type | Value |
   | --- | --- | --- |
   | `GESTUREFLOW_ENV` | variable | `production` |
   | `GESTUREFLOW_DATABASE_URL` | **secret** | the Neon string from step 1 |
   | `GESTUREFLOW_JWT_SECRET` | **secret** | output of `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
   | `GESTUREFLOW_SENTENCE_PROVIDER` | variable | `groq` |
   | `GESTUREFLOW_GROQ_API_KEY` | **secret** | the Groq key from step 2 |
   | `GESTUREFLOW_CORS_ORIGINS` | variable | `["http://localhost:3000"]` for now; updated in step 5 |

3. Let GitHub deploy to it. Create a token on huggingface.co → **Settings → Access Tokens →
   New token**, type **Write**. Then in the GitHub repo → **Settings → Secrets and variables →
   Actions**:
   - **Secrets** tab: `HF_TOKEN` = that token
   - **Variables** tab: `HF_SPACE` = `your-hf-username/gestureflow-api`
4. GitHub → **Actions → Deploy API → Run workflow**. From now on it also runs on every push
   to `main`.
5. In the Space, the **Logs** tab shows the build installing requirements (5–10 minutes the
   first time, mostly PyTorch), then `Uvicorn running on http://0.0.0.0:7860`. Open the Space's app URL, `https://<you>-gestureflow-api.hf.space`.
   It should show the API docs, and `/api/v1/health` should say `"status":"ok"`,
   `"database":"ok"`, `"sentence_provider":"groq"`.

## 4. Web app — Vercel

1. Sign up at vercel.com with GitHub → **Add New → Project** → import this repository.
2. Settings on the import screen:
   - **Root Directory**: `frontend`
   - Framework: Next.js (detected)
   - **Environment Variable**: `NEXT_PUBLIC_API_URL` = your Space URL from step 3, with no
     trailing slash
3. Deploy. The build copies the Edge-mode model from `models/`, outside `frontend/`. That
   works because Vercel includes files outside the Root Directory by default. If the build
   log says `gesture_cnn.onnx not found`, turn on **Settings → General → Root Directory →
   Include files outside the root directory** and redeploy.

## 5. Connect them

1. Copy your Vercel URL, e.g. `https://gestureflow.vercel.app`.
2. In the Hugging Face Space settings, set `GESTUREFLOW_CORS_ORIGINS` to
   `["https://gestureflow.vercel.app"]`. Add `"http://localhost:3000"` to the list too if you
   still want local development against the live API. Restart the Space.

## 6. Check it works

- The Vercel URL loads, and **Try it now** works without an account (Edge mode).
- **Create an account**, then sign in.
- **Recognize → Server**: letters commit, and the session shows on the Dashboard.
- **Make a sentence** returns a sentence. That's Groq answering.
- **Settings → Delete account** removes it, and signing in again fails.

If something fails, copy the browser console error and the Space's **Logs** tab output to
Claude.

---

## Using PostgreSQL locally instead of SQLite

SQLite is the zero-setup default. To develop against PostgreSQL, the same database type as
production, use the PostgreSQL 18 already installed on your PC.

1. Create a database and user. This asks for the `postgres` password you set when installing
   PostgreSQL; pick your own password for the new user:
   ```powershell
   & "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -c "CREATE USER gestureflow WITH PASSWORD 'choose-one';" -c "CREATE DATABASE gestureflow OWNER gestureflow;"
   ```
2. In `.env`, in the project folder:
   ```
   GESTUREFLOW_DATABASE_URL=postgresql://gestureflow:choose-one@localhost:5432/gestureflow
   ```
3. Create the tables, then start the API as usual:
   ```powershell
   uv run alembic upgrade head
   ```

Delete or comment out that line in `.env` to go back to SQLite. The two databases are
separate; nothing is copied between them.
