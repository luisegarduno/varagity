# To build this image : docker build -t varagity-app . --progress=plain
# To run:             : docker run --rm varagity-app

# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Tesseract for the pluggable OCR fallback: both benchmark engines must be
# available in-container (EasyOCR arrives as a pip dependency of docling).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

# Setup a non-root user
RUN groupadd --system --gid 999 user \
 && useradd --system --gid 999 --uid 999 --create-home user

# Pre-create the model cache mount point owned by the app user, so the named
# volume (compose: model_cache:/home/user/.cache) inherits writable ownership.
RUN mkdir -p /home/user/.cache && chown -R user:user /home/user/.cache

# Copy the project into the image
ADD . /app

# Install the project into `/app`
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Use the non-root user to run our application
USER user

# Run main.py
CMD ["uv", "run", "main.py"]
