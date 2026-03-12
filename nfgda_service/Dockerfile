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

RUN pip install --no-cache-dir GDAL==$(gdal-config --version)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "main.py"]