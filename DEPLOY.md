# Deploying DocMind

Recommended: a single small VPS running Docker Compose behind Caddy (automatic
HTTPS), serving the whole app from one domain. DocMind is stateful (FAISS index,
uploads, users, jobs) and runs always-on background workers, so a persistent
container host fits far better than serverless.

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
