# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# FIX: Added 'git' (for git-based pip packages) and 'dos2unix' (to fix Windows file encodings)
# 'libgl1' and 'libglib2.0-0' are required for OpenCV/image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    dos2unix \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .

# FIX: 
# 1. Run dos2unix to convert Windows (CRLF/UTF-16) formatting to Linux (LF/UTF-8)
# 2. Robustly remove Windows-only packages
# 3. Upgrade pip and install dependencies
RUN dos2unix requirements.txt && \
    grep -vEi "pywin32|pypiwin32|win32" requirements.txt > requirements.tmp && \
    mv requirements.tmp requirements.txt && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# FIX: Use 'python -m uvicorn' to ensure the module path is resolved correctly
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]