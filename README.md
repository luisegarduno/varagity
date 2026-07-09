# Varagity

Full Stack RAG application

> 📄 See [`spec.md`](spec.md) for the full system design & architecture.


## ToDo

Project ToDo's:
- [ ] Create pre-commit rules

Stack ToDo's:
- [ ] Add: PostgreSQL + pgvector
- [ ] Add: Elasticsearch (contextual BM25)
- [ ] Add: llama.cpp server (self-hosted LLM)
- [ ] Add: Prefect
- [ ] Add: Prefect-Prometheus-Exporter


RAG ToDo's:
- [ ] Model Loader
- [ ] Contextual Embeddings
- [ ] Contextual BM25
- [ ] Re-ranker



-----------------------------

# Instructions

## Pre-Req's

*  \>=12GB's of disk space
* `Nvidia-Docker` must be setup
    <details><summary>Instructions (debian)</summary>

    1. Add the package repositories (modern method without apt-key)
        ```bash
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        ```

    2. Install nvidia-container-toolkit
        ```bash
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        ```

    3. Configure Docker to use the runtime
        ```bash
        sudo nvidia-ctk runtime configure --runtime=docker
        ```

    4. Lastly, restart docker daemon
        ```bash
        sudo systemctl restart docker
        ```
    </details>

## Running Varagity

1. Build & Run:
    ```bash
    docker compose up
    ```

2. Done! Navigate to [localhost:8081/v1/docs](http://localhost:8081/v1/docs), to checkout
the FastAPI endpoints!
    * Below is an example of how you can use `/v1/embeddings` :
    ```bash
    curl -X 'POST' \
      'http://0.0.0.0:8081/v1/embeddings' \
      -H 'accept: application/json' \
      -H 'Authorization: Bearer YOUR_INFINITY_API_KEY_HERE' \
      -H 'Content-Type: application/json' \
      -d '{
      "model": "infloat/multilingual-e5-large-instruct",
      "encoding_format": "float",
      "user": "string",
      "dimensions": 0,
      "input": [
        "This is a sample sentence!"
      ],
      "modality": "text"
    }'
    ```

---------------
