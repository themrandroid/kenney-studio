# # Use a lightweight Python base image
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Install OpenCV and system dependencies
# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     && rm -rf /var/lib/apt/lists/*

# # Copy all files
# COPY . /app

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Streamlit-specific settings
# ENV PORT=7860
# EXPOSE 7860

# # Fix cache folder permissions
# RUN mkdir -p /app/.cache/huggingface /app/.streamlit && \
#     chmod -R 777 /app/.cache /app/.streamlit

# # Environment vars for writable paths
# ENV HF_HOME=/app/.cache/huggingface
# ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
# ENV STREAMLIT_HOME=/app/.streamlit

# # Run Streamlit app
# CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]\

# Base image with Streamlit
FROM python:3.10-slim

# Set working directory
WORKDIR /app
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y git && apt-get clean
RUN pip install --no-cache-dir -r requirements.txt

# Fix Streamlit permission issue
ENV HOME=/app
RUN mkdir -p /app/.streamlit && chmod -R 777 /app/.streamlit
RUN mkdir -p /app/temp && chmod -R 777 /app /tmp

EXPOSE 7860

# Tell Streamlit where to store config explicitly
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]