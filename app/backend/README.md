# LDMAX FastAPI Backend

A high-performance FastAPI server designed to serve pretrained **Diffusion Transformer (DiT)** checkpoints from Google Cloud Storage (`gs://diffjax/...`) or local disk, ready for deployment on **Google Cloud Run**.

---

## ⚡ API Endpoints

- `GET /api/health` — Returns backend health status and active JAX compute devices (`cpu`, `gpu`, or `tpu`).
- `GET /api/metadata` — Returns metadata for all datasets, classes, and 40 CelebA attributes.
- `POST /api/generate/cifar10` — Generates $32 \times 32$ RGB images with class weight blending.
- `POST /api/generate/fashion_mnist` — Generates $28 \times 28$ Grayscale images with class weight blending.
- `POST /api/generate/celeba` — Generates $256 \times 256$ RGB images via Latent Diffusion and VAE decoding.
- `POST /api/generate` — Unified generation endpoint.

---

## 💻 Local Execution

```bash
# 1. Activate conda environment
conda activate ldmax

# 2. Run backend server
PYTHONPATH=. uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger docs will be available at: `http://localhost:8000/docs`.

---

## ☁️ Google Cloud Run Deployment Guide

### Prerequisites
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk) installed and authenticated.
- Target GCP project set (`gcloud config set project YOUR_PROJECT_ID`).

### 1. Build and Submit Container Image

```bash
gcloud builds submit \
    --tag gcr.io/YOUR_PROJECT_ID/ldmax-backend \
    --file app/backend/Dockerfile .
```

### 2. Deploy to Cloud Run

```bash
gcloud run deploy ldmax-backend \
    --image gcr.io/YOUR_PROJECT_ID/ldmax-backend \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --allow-unauthenticated \
    --set-env-vars CORS_ORIGINS="https://ghif.github.io,http://localhost:3000"
```

Once deployed, Cloud Run will output a public service URL (e.g. `https://ldmax-backend-ejw7j7q5fa-uc.a.run.app`). Enter this URL into the frontend settings modal to connect your GitHub Pages deployment!
