FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Run as an unprivileged user; only data/ (SQLite) and .cache/ (FAISS index) need to be writable.
RUN useradd --create-home appuser && mkdir -p data .cache && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
