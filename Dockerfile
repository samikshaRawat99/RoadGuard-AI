FROM python:3.11-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS libraries required by OpenCV and video processing (ffmpeg, libgl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO11n weights so startup is instant
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Copy application source code
COPY . .

# Ensure outputs directories exist
RUN mkdir -p outputs/uploads outputs/processed_videos

# Expose port (Render/Railway injects $PORT)
EXPOSE 8000

# Start command
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
