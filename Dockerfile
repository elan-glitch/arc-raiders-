FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg and build deps
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages — no torch, no whisper (keeps build fast)
RUN pip install --no-cache-dir \
    flask \
    flask-cors \
    yt-dlp \
    librosa \
    numpy \
    scipy \
    opencv-python-headless

# Copy app
COPY . .

EXPOSE 5000

CMD ["python3", "app.py"]
