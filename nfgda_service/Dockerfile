FROM python:3.11-slim

WORKDIR /app

# Install build tools and libraries required by cartopy, arm_pyart, and gdal
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    make \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libhdf5-dev \
    pkg-config \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache dependency installs
COPY requirements.txt /app/requirements.txt

# Pin Python GDAL to the system GDAL version reported by gdal-config
RUN echo "GDAL==$(gdal-config --version)" > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt --no-binary=GDAL GDAL==$(gdal-config --version) && \
    pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

COPY . /app

ENV PYTHONPATH="/app/algorithm"

CMD ["python", "main.py"]