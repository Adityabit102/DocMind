# Deploying DocMind

Two supported paths:
- **[Free: Hugging Face Spaces + Vercel](#free-hugging-face-spaces--vercel)** —
  backend on a free 16 GB Space, frontend on Vercel, state persisted to a free
  private HF dataset. No card required.
- **[Paid/own server: VPS + Docker Compose + Caddy](#vps--docker-compose--caddy)**
  — everything on one box behind automatic HTTPS.

DocMind is stateful (FAISS index, uploads, users, jobs) and runs always-on
background workers, so it needs a persistent, non-serverless backend.

---

# Free: Hugging Face Spaces + Vercel

Backend → a free HF Space (Docker, 2 vCPU / 16 GB RAM). Frontend → Vercel.
Persistence → the `data/` dir is synced to a **private HF dataset** on every
change and on shutdown, and restored on startup (free Spaces have ephemeral
disk, so this is what keeps your index between restarts). Generation runs on
Groq, so no GPU is needed. **No features are lost.**

### 1. Create a HuggingFace access token
huggingface.co → Settings → **Access Tokens** → **New token**, role **Write**.
Copy it (starts with `hf_...`). The app auto-creates the dataset on first push.

### 2. Create the Space
huggingface.co → **New Space** → SDK **Docker**, **Blank**, visibility your
choice. Then push this repo to it:
```bash
git remote add space https://huggingface.co/spaces/<your-hf-username>/docmind
git push space main
```
(The `README.md` already carries the Space's Docker config in its frontmatter.)

### 3. Set Space secrets
In the Space → **Settings → Variables and secrets**, add:
```
GROQ_API_KEY      = gsk_your_fresh_key
LLM_PROVIDER      = groq
LLM_MODEL         = llama-3.3-70b-versatile
HF_TOKEN          = hf_your_write_token
HF_DATASET_REPO   = <your-hf-username>/docmind-data
CORS_ORIGINS      = https://<your-app>.vercel.app
```
The Space builds the Dockerfile and boots on `https://<user>-docmind.hf.space`.
First build takes a few minutes (it bakes the models in).

### 4. Deploy the frontend on Vercel
vercel.com → **Add New Project** → import this GitHub repo:
- **Root Directory:** `frontend`
- **Environment Variable:** `NEXT_PUBLIC_API_URL = https://<user>-docmind.hf.space`
- Deploy. You get `https://<your-app>.vercel.app`.

### 5. Connect the two
Set the Space's `CORS_ORIGINS` to your final Vercel URL (step 3), and confirm
the Vercel `NEXT_PUBLIC_API_URL` points at the Space. Redeploy whichever you
changed. Open the Vercel URL — done.

### Verify
```bash
curl https://<user>-docmind.hf.space/api/v1/health      # {"status":"ok",...}
```
Upload a doc, then restart the Space (Settings → Factory reboot) — your
documents should still be there, restored from the HF dataset.

### Notes
- The free Space **sleeps after ~48 h idle** and wakes on the next visit
  (state is restored from the dataset, so nothing is lost).
- Conversations are stored in your browser (localStorage), so chat history
  survives regardless of the backend.

---

# VPS + Docker Compose + Caddy

Everything on one box behind automatic HTTPS — best if you want a custom domain
and 24/7 uptime without sleeps.

## 1. Provision a server
- Any VPS with **≥ 4 GB RAM**, 2 vCPU, ~20 GB disk, Ubuntu 24.04.
  (Hetzner CX22 ≈ €4/mo, DigitalOcean/Lightsail ≈ $12/mo.)
- Point a **DNS A record** for your domain (e.g. `docmind.example.com`) at the
  server's public IP. Open ports **80** and **443**.

## 2. Install Docker
```bash
curl -fsSL https://get.docker.com | sh
```

## 3. Get the code + configure
```bash
git clone https://github.com/Adityabit102/DocMind.git
cd DocMind
cp .env.example .env
```
Edit `.env` and set at minimum:
```
DOMAIN=docmind.example.com
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_fresh_key          # rotate the old one!
LLM_MODEL=llama-3.3-70b-versatile
CORS_ORIGINS=https://docmind.example.com
# If you enable accounts later:
# ENABLE_AUTH=true
# AUTH_SECRET=<a long random string>
```

## 4. Launch (backend + frontend + redis + Caddy)
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```
Caddy provisions a TLS cert automatically on first run. Give it ~30s, then open
`https://docmind.example.com`.

## 5. Verify
```bash
curl https://docmind.example.com/api/v1/health      # {"status":"ok",...}
docker compose ps                                    # all services "Up"
docker compose logs -f backend                       # watch startup
```

## Operating it
- **Update to latest:** `git pull && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build`
- **Logs:** `docker compose logs -f backend` (or `frontend` / `caddy`)
- **Backups:** the `data/` directory (FAISS index, uploads, metadata) and the
  `redis_data` volume are the only state — snapshot the server disk or
  `tar czf backup.tgz data/`.
- **Stop:** `docker compose -f docker-compose.yml -f docker-compose.prod.yml down`

## Notes
- Embeddings + reranker are baked into the image, so no model download at
  runtime and no cold-start delay.
- Generation uses Groq (external API) — no GPU required on the server.
- Redis is optional; the app falls back to an in-memory cache if it's absent.
  To use it, set `CACHE_BACKEND=redis` and `REDIS_URL=redis://redis:6379`.
- Optional heavy integrations (RAGAS, Presidio, unstructured, evidently) are
  off by default with graceful fallbacks; enable them in `requirements.txt`
  if you need them.

## Alternative: managed (no server admin)
- **Frontend → Vercel**: import the repo, set root to `frontend/`, add build env
  `NEXT_PUBLIC_API_URL=https://<your-backend-domain>`.
- **Backend → Render/Railway**: deploy the `Dockerfile` as a web service with a
  **persistent disk mounted at `/app/data`** and a paid (always-on) instance —
  free/scale-to-zero tiers will drop the index and re-download models. Set the
  same env vars as above plus `CORS_ORIGINS=https://<your-vercel-domain>`.
