# Deployment Guide

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Streamlit Cloud (Recommended - Free)

1. Push your code to GitHub
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file to `app.py`
7. Click "Deploy"

Your app will be live at `https://your-app-name.streamlit.app`

## Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t glucose-mlp-app .
docker run -p 8501:8501 glucose-mlp-app
```

## Heroku Deployment

Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]" > ~/.streamlit/config.toml
echo "port = $PORT" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
```

Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## Environment Variables

Set these if needed:
- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)
