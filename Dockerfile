# To build this image : docker build -t varag . --progress=plain
# To run:             : docker run --rm varag  

# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Setup a non-root user
RUN groupadd --system --gid 999 user \
 && useradd --system --gid 999 --uid 999 --create-home user

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
