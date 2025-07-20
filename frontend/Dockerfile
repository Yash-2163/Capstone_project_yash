FROM python:3.10-slim

# Install system-level dependencies required for building certain Python packages
# (e.g., psycopg2-binary needs libpq-dev). This prevents common installation errors.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first and install dependencies.
# This leverages Docker's layer caching. The dependencies will only be re-installed
# if the requirements.txt file changes, making subsequent builds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port that the Flask application will run on
EXPOSE 5000

# The command to run your Flask application when the container starts
CMD ["python", "app.py"]
