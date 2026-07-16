# To build this image : docker build -t varagity-app . --progress=plain
# To run:             : docker run --rm varagity-app

# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Tesseract for the pluggable OCR fallback: both benchmark engines must be
# available in-container (EasyOCR arrives as a pip dependency of docling).
# libgl1: cv2's native module links libGL — rapidocr (a docling core dep)
# ships non-headless opencv-python, which clobbers easyocr's headless build
# in the shared cv2/ directory. Docling's TableFormer and EasyOCR both
# `import cv2`, so without libGL every PDF conversion dies at pipeline init
# (and docling masks the real error as "EasyOCR is not installed").
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Setup a non-root user. uid/gid default to 1000 to match the host user —
# this image shares the model_cache volume (and the ./docs mount) with the
# api image, so both must agree on ownership (see Dockerfile.api for the
# rationale and the build-arg override).
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid ${APP_GID} user \
 && useradd --gid ${APP_GID} --uid ${APP_UID} --create-home user

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
