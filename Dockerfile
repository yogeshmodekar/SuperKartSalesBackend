
# ============================================================
# SuperKart Sales Prediction API - Docker Configuration
# ============================================================

# Use a lightweight Python base image
# This provides Python runtime environment required for the Flask API
FROM python:3.9-slim


# ============================================================
# Set Working Directory
# ============================================================

# Define the working directory inside the Docker container
# All application files and commands will execute from this location
WORKDIR /app


# ============================================================
# Copy Application Files
# ============================================================

# Copy all project files from the local directory into the container
# This includes:
# - Flask API application (superkart_sales_app.py)
# - Trained ML model (.joblib)
# - requirements.txt
COPY . .


# ============================================================
# Install Python Dependencies
# ============================================================

# Install all required Python packages listed in requirements.txt
# --no-cache-dir avoids storing pip cache files and helps reduce image size
RUN pip install --no-cache-dir --upgrade -r requirements.txt


# ============================================================
# Start Flask API Using Gunicorn
# ============================================================

# Launch the production WSGI server using Gunicorn
#
# Parameters:
# -w 4:
#   Starts 4 worker processes to handle multiple API requests concurrently
#
# -b 0.0.0.0:7860:
#   Binds the application to port 7860 and makes it accessible externally
#
# superkart_sales_app:superkart_api:
#   Refers to:
#       superkart_sales_app.py  -> Python application file
#       superkart_api           -> Flask application instance
#
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "superkart_sales_app:superkart_api"]
