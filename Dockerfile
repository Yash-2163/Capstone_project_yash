FROM apache/airflow:2.9.1-python3.10

USER root

# Install the OpenMP runtime so LightGBM can load
RUN apt-get update \
 && apt-get install -y libgomp1 \
 && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --default-timeout=120 --retries=5 --no-cache-dir -r /requirements.txt
