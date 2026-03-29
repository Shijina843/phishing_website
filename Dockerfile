# Use an official Python light runtime as a parent image
FROM python:3.12-slim

# Prevent Python from writing pyc files to disk and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and set the working directory
WORKDIR /app

# Install system dependencies (needed for compiling some ML libraries if wheels aren't matched)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies defined in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port (Render defaults to 10000, Flask usually 5000, but we bind 0.0.0.0 directly)
EXPOSE 5000

# Run the app using gunicorn (Production-ready web server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
