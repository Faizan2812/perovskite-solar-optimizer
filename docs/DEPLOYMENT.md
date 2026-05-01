# Deployment Guide

How to run this tool locally, on Streamlit Community Cloud, or on your own
server.

## Local

```bash
git clone https://github.com/Faizan2812/perovskite-solar-optimizer.git
cd perovskite-solar-optimizer
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

### Troubleshooting local installs

**JAX wheel fails on Windows**: JAX on Windows is historically flaky. The
main PINN (`ai/pinn_real.py`) uses PyTorch and does not need JAX. If you
don't need the Poisson-only JAX PINN, comment out `jax` and `jaxlib` in
`requirements.txt`.

**PyTorch CUDA vs CPU**: the default `torch==2.2.0` in requirements is the
CPU build. For GPU, install the CUDA wheel manually:
```bash
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

**Streamlit not found after install**: check you're in the right venv. Run
`which streamlit` (Linux/macOS) or `where streamlit` (Windows).

---

## Streamlit Community Cloud (recommended for sharing)

Streamlit Community Cloud gives you a free public URL and automatic redeploy
on every git push.

1. Push this repo to GitHub (public or private).
2. Sign in at https://streamlit.io/cloud with your GitHub account.
3. Click **New app**. Point it at:
   - Repository: `Faizan2812/perovskite-solar-optimizer`
   - Branch: `main`
   - Main file: `app.py`
4. Click **Deploy**. First build takes 3-5 minutes (pip install).

### Known Streamlit Cloud gotchas

**Free tier has 1 GB RAM**. Training the PINN or running DD benchmarks inside
the Streamlit process will hit OOM. For this reason, the app buttons for
PINN training and BO/NSGA runs show an info message telling users to run
those from the CLI instead. This is intentional.

**JAX can fail to install** on the free tier — it requires a newer glibc than
Streamlit Cloud ships. Comment out `jax` and `jaxlib` if the build fails;
the main PINN doesn't need them.

**App sleeps after 7 days of inactivity**. Anyone visiting the URL will wake
it up in about 30 seconds.

### secrets

If you need API keys or credentials, put them in `.streamlit/secrets.toml`.
Do NOT commit that file — it's already in `.gitignore`. Streamlit Cloud
exposes a GUI for entering them.

---

## Self-hosting on your own server

Any machine with Python 3.10+ and 2 GB RAM works.

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Put it behind a reverse proxy (nginx) for production. A minimal nginx config:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Docker

A minimal Dockerfile (not shipped; add if useful):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t psc-tool .
docker run -p 8501:8501 psc-tool
```
