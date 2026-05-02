# Lightweight Python
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Python settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI backend (NOT Streamlit)
CMD ["python", "-m", "uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]