# LDMAX Web Frontend

A responsive web frontend for interacting with Diffusion Transformers in JAX. Designed to be deployed on **GitHub Pages** (at `ghif.github.io/ldmax`) and connect to the FastAPI backend running on **Google Cloud Run**.

---

## 🌟 Features

- **Multi-Dataset Tabs**:
  - **CIFAR-10** ($32 \times 32$ RGB raw-pixel DiT) with 10 class influence blending sliders and quick presets.
  - **Fashion-MNIST** ($28 \times 28$ Grayscale raw-pixel DiT) with class influence blending sliders and quick presets.
  - **CelebA** ($256 \times 256$ RGB Latent DiT + VAE) with 40 searchable facial attribute toggles and combo presets.
- **Client-side Interactivity**:
  - Real-time backend connectivity and platform indicator (CPU/TPU).
  - Dynamic API URL configuration modal with local storage persistence.
  - Image lightbox with zoom, individual PNG download, and latency metrics.
  - Dark / Light mode toggle with local storage persistence.
  - Zero heavy external JavaScript dependencies (vanilla modern ES6+).

---

## 🚀 GitHub Pages Deployment Guide

### Option 1: GitHub Actions (Recommended)

Create a workflow file at `.github/workflows/deploy-pages.yml`:

```yaml
name: Deploy Frontend to GitHub Pages

on:
  push:
    branches:
      - main
    paths:
      - 'app/frontend/**'

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: 'pages'
  cancel-in-progress: true

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: 'app/frontend'

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### Option 2: Deploy to `gh-pages` branch

```bash
# From repository root
git subtree push --prefix app/frontend origin gh-pages
```

---

## 💻 Local Development

You can serve the frontend with any static file server:

```bash
# Using Python built-in HTTP server
cd app/frontend
python3 -m http.server 3000
```

Open your browser at `http://localhost:3000`. By default, it will automatically look for the FastAPI backend at `http://127.0.0.1:8000`.
