# Deployment Guide

## Overview
- **Backend (FastAPI + RAG)** → Render.com (Docker deployment)
- **Frontend (Static files)** → GitHub Pages (automated via GitHub Actions)

---

## Part 1: Deploy Backend to Render

### Step 1: Push to GitHub
```bash
git add -A
git commit -m "ready for deployment"
git push origin main
```

### Step 2: Create Render Web Service
1. Go to https://render.com and sign up/login
2. Click **New → Web Service**
3. Connect your GitHub account and select this repository
4. Configure:
   - **Name**: `k8s-copilot` (or any name)
   - **Environment**: `Docker`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Docker Build Context**: `.` (root)

### Step 3: Set Environment Variables
In the Render dashboard, add:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   MODEL_NAME=llama-3.3-70b-versatile
   EMBEDDING_DEVICE=cpu
   ```

### Step 4: Deploy
- Click **Create Web Service**
- Wait ~5-10 minutes for build
- Copy your Render URL: `https://your-app-name.onrender.com`

### Step 5: Ingest Sample Data
After deployment completes, run this once:
```bash
curl -X POST https://your-app-name.onrender.com/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "data/sample", "no_drop": false}'
```

✅ **Backend is live!** Test: `https://your-app-name.onrender.com/health`

---

## Part 2: Deploy Frontend to GitHub Pages

### Step 1: Update API URL in config.js
Edit `static/config.js` and replace the placeholder:
```javascript
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? ''  // Local: use same origin
  : 'https://your-app-name.onrender.com';  // ← PUT YOUR RENDER URL HERE
```

### Step 2: Commit the Change
```bash
git add static/config.js
git commit -m "config: set Render backend URL"
git push origin main
```

### Step 3: Enable GitHub Pages
1. Go to your repo on GitHub
2. Navigate to **Settings → Pages**
3. Under **Build and deployment**:
   - Source: `GitHub Actions`
4. The workflow (`.github/workflows/deploy-pages.yml`) will automatically run

### Step 4: Wait for Deployment
- Go to **Actions** tab in your repo
- Watch the "Deploy Static Frontend to GitHub Pages" workflow
- Takes ~1-2 minutes
- Your site will be live at: `https://your-username.github.io/repo-name/`

✅ **Frontend is live!** Visit the URL and test the chat interface.

---

## What Gets Deployed Where

### To Render (Backend):
- `src/` - Python backend code
- `static/` - Served by FastAPI (not used for GitHub Pages)
- `templates/` - FastAPI templates (not used for GitHub Pages)
- `data/` - Sample data for ingestion
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container build instructions

### To GitHub Pages (Frontend only):
- `static/app.js` - 3D UI and chat logic
- `static/config.js` - API endpoint configuration
- `static/styles.css` - Styling
- `index.html` - Main HTML file

**Result**: Only ~1 MB of static files deployed to GitHub Pages (fast!)

---

## Important Notes

### Free Tier Limitations

**Render Free Tier:**
- Spins down after 15 minutes of inactivity
- First request after sleep takes ~30 seconds (cold start)
- No persistent disk included (uses `/tmp/` storage — resets on restart)
- 750 hours/month free

**GitHub Pages:**
- 1 GB storage limit
- 100 GB bandwidth/month
- Public repos only (for free tier)

### Data Persistence

With `/tmp/` storage on Render:
- ✅ Chroma vector store persists during app lifetime
- ✅ Chat memory persists during app lifetime
- ❌ Both are wiped on restart/redeploy

**To get persistent storage on Render:**
- Upgrade to paid plan ($7/month)
- Add a disk, mount at `/data`
- Update `.env`: `CHROMA_PERSIST_DIR=/data/chroma` and `MEMORY_PATH=/data/chat_memory.json`
- Update `Dockerfile` ENV vars to match

---

## Testing Locally Before Deploy

```bash
# Test backend
docker compose up -d
curl http://localhost:8000/health

# Test frontend
open http://localhost:8000
```

---

## Troubleshooting

### Backend won't start on Render
- Check **Logs** in Render dashboard
- Verify `GROQ_API_KEY` is set correctly
- Ensure Dockerfile builds successfully locally first

### Frontend can't connect to backend
- Check `static/config.js` has correct Render URL
- Open browser DevTools → Network tab
- Look for CORS errors (should be fixed, but verify)

### "No documents found" in RAG responses
- Run the ingest curl command (Step 5 of Part 1)
- Check `/ingest` endpoint returns `chunks_stored > 0`

### GitHub Pages shows 404
- Verify workflow ran successfully in Actions tab
- Check Settings → Pages shows the correct source
- Wait 2-3 minutes after workflow completes

---

## Cost Breakdown

| Service | What | Cost |
|---------|------|------|
| Render (Free) | Backend hosting | $0 |
| GitHub Pages | Frontend hosting | $0 |
| Groq API | LLM inference | $0 (free tier) |
| **Total** | | **$0/month** |

For production with persistent storage: ~$7/month (Render disk)
