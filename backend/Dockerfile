# Full python app image: conform this to GDAL requirements, or run secondary venv for 3.11? Hmm
FROM python:3.14.2

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8001
EXPOSE 8001

# Install gunicorn
RUN pip install gunicorn

# Use gunicorn to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "app:app", "--workers", "4"]
