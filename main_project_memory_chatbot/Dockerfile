FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port FastAPI runs on
EXPOSE 7860

# Run the FastAPI server on Hugging Face's default port (7860)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
