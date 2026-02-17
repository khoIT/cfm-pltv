# Deployment Guide for CFM pLTV Streamlit App

## ⚠️ Important: Netlify is NOT Compatible

**Netlify does not support Streamlit applications** because:
- Streamlit requires a persistent Python server process
- Netlify is designed for static sites and serverless functions only
- Streamlit apps need WebSocket connections for real-time updates

## ✅ Recommended Deployment Options

### 1. **Streamlit Community Cloud** (FREE & Easiest)

**Steps:**
1. Push your code to GitHub (public or private repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repo, branch, and main file path: `webapp/app.py`
6. Click "Deploy"

**Requirements:**
- `requirements.txt` ✅ (already exists)
- `.streamlit/config.toml` ✅ (created)
- Data files must be in the repo or use `st.file_uploader()`

**Limitations:**
- Free tier: 1GB RAM, limited CPU
- Data files in repo should be < 100MB
- For large datasets, use external storage (S3, GCS) or file upload

---

### 2. **Heroku** (Paid, More Resources)

**Steps:**
1. Create `Procfile`:
   ```
   web: streamlit run webapp/app.py --server.port=$PORT
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. Deploy:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

---

### 3. **AWS EC2 / Google Cloud VM** (Full Control)

**Steps:**
1. Launch a VM instance
2. Install Python 3.9+
3. Clone your repo
4. Install dependencies: `pip install -r requirements.txt`
5. Run with PM2 or systemd:
   ```bash
   streamlit run webapp/app.py --server.port=8501
   ```
6. Configure nginx reverse proxy for HTTPS

---

### 4. **Docker + Cloud Run / ECS** (Containerized)

**Dockerfile:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY webapp/requirements.txt .
RUN pip install -r requirements.txt
COPY webapp/ .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Deploy to Google Cloud Run:**
```bash
gcloud run deploy cfm-pltv --source . --platform managed --region us-central1 --allow-unauthenticated
```

---

## 📦 Data Handling for Production

Your app currently loads data from local CSV files. For production:

### Option A: Include Sample Data (< 100MB)
- Keep small sample datasets in the repo
- Add to `.gitignore` exceptions if needed

### Option B: File Upload Widget
```python
uploaded_file = st.file_uploader("Upload training data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
```

### Option C: Cloud Storage
```python
import boto3
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='your-bucket', Key='cfm_pltv.csv')
df = pd.read_csv(obj['Body'])
```

---

## 🔐 Secrets Management

For API keys, database credentials, etc., use Streamlit secrets:

**Local:** Create `webapp/.streamlit/secrets.toml`
```toml
[aws]
access_key = "YOUR_KEY"
secret_key = "YOUR_SECRET"
```

**Streamlit Cloud:** Add secrets in the app settings dashboard

**Access in code:**
```python
import streamlit as st
aws_key = st.secrets["aws"]["access_key"]
```

---

## 📊 Current App Structure

```
webapp/
├── app.py                 # Main entry point
├── shared.py              # Shared utilities
├── pages/                 # Multi-page app
│   ├── 1_Decision_Definition.py
│   ├── 2_Features_and_Model.py
│   ├── 3_Evaluation_and_Insights.py
│   ├── 4_Action_and_Simulation.py
│   ├── 5_Feedback_and_Learning.py
│   └── 6_Diagnostics.py
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
└── .gitignore             # Files to exclude from git
```

---

## 🚀 Quick Start: Deploy to Streamlit Cloud NOW

1. **Initialize git repo (if not already):**
   ```bash
   cd C:\Users\CPU12830-local\code\cfm_pltv
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Create GitHub repo and push:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/cfm-pltv.git
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select repo: `YOUR_USERNAME/cfm-pltv`
   - Main file path: `webapp/app.py`
   - Click "Deploy"

4. **Done!** Your app will be live at `https://YOUR_USERNAME-cfm-pltv.streamlit.app`

---

## 📝 Notes

- The `netlify.toml` file created earlier is **not usable** for this Streamlit app
- For production, ensure data files are either:
  - Included in repo (< 100MB)
  - Uploaded via UI
  - Fetched from cloud storage
- Monitor resource usage on free tier (1GB RAM limit)
