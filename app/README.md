# LDMAX Full-Stack Web Application

This directory contains the production-ready full-stack application for **LDMAX**, replicating the functionality of the interactive `scripts/demo.py` GUI across a decoupled frontend and backend.

```text
app/
├── frontend/             # Static web app (Deployable to GitHub Pages: ghif.github.io/ldmax)
│   ├── index.html        # Semantic HTML5 tabbed interface
│   ├── css/
│   │   └── style.css     # Modern, responsive glassmorphic styles
│   ├── js/
│   │   └── app.js        # Vanilla JS API client, tab switching, and base64 image rendering
│   └── README.md         # GitHub Pages deployment instructions
│
└── backend/              # FastAPI server (Deployable to Google Cloud Run)
    ├── main.py           # FastAPI routes, CORS, and request schemas
    ├── service.py        # Model cache, DDIM sampling engine, and VAE decoding
    ├── config.py         # Configuration settings & environment variables
    ├── requirements.txt  # Python server dependencies
    ├── Dockerfile        # Container image for Cloud Run
    └── README.md         # Cloud Run deployment instructions
```

---

## ⚡ Quickstart (Local Full-Stack Run)

### 1. Launch FastAPI Backend
```bash
conda activate ldmax
PYTHONPATH=. uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Launch Static Frontend
```bash
python3 -m http.server 3000 --directory app/frontend
```

Open `http://localhost:3000` in your browser. The frontend will automatically connect to `http://127.0.0.1:8000`.
