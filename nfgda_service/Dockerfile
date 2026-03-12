FROM python:3.11-slim

WORKDIR /app

# Install build tools and libraries required by cartopy and arm_pyart
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
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "main.py"]