
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set thư mục làm việc
WORKDIR /app

# Copy toàn bộ project vào Docker container
COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "try:app", "--host", "0.0.0.0", "--port", "8000"]
