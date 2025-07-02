# Use a slim Python image
FROM python:3.11-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Start FastAPI
CMD ["fastapi", "run", "main.py", "--port", "8000"]
