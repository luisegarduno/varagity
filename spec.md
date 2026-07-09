Alright so i'm planning on creating a fullstack RAG project

This will be a complex system in the future so it needs to be structured correctly and must be modular.

------------------------------------

Here is a basic overview of a vanilla RAG system:
1. Pre-process documents (extract contents from file)
2. Split corpus of documents into chunks
3. Embed chunks into vectors
4. Put vectors into a vector store index
5. Create (context) Prompt for LLM (This prompt tells the LLM to answer query's given context's found in the search step)

then at runtime

6. User provides a query to LLM
7. User query is vectorized with same encoder model
8. Execute search this query vector against the vector store index
9. Find top-k results
10. Retrieve corresponding chunks from database
11. Feed chunks into LLM prompt as context

------------------------------------------------------

Tools/Technologies:
- The entire project will be built using docker compose
- Most of the project will be built using Python (uv) - only the frontend will be built using typescript
- Embedding models will be hosted using infinity embeddings
- We'll use PostgreSQL to store our vectors

------------------------------------------------------

For now we'll build out a simple version:
- No ui for now, terminal based app for now

The docker container will start up the following:
* llama.cpp server (hosting llm)
* infity embedding server (hosting embedding model)
* PostgreSQL vector database
* Prefect

The python app:
1. After starting: The app checks the "docs" directory to see if there are documents in there
2. If yes, we will retrieve the full filepath for each of the files (for now we'll only allow PDF, text, markdown extraction - ignore all other files)
3. We'll then store the filepaths in an array depending on the file's filetype extension - text & markdown will go in the same array (since text extraction is the same), PDF's will go in a different array (since we need to use docling to extract the contents)
4. Based on the file type, we'll then send that array to a specific function to then extract the contents of it. 
5. Once I have a file loaded (aka once we've extracted the contents of it) - then we proceed to chunk it and save them in an array
6. Embed the chunks using an embedding model - this should just be calling the infinity embedding api
7. Append the embedded chunks (which includes the metadata) to the vector database


LOOK AT THE CODE IN /home/blurry/Desktop/ML/RAG-Research/Demos/Demo-ContextualRetrieval - as for the most part the logic is correct. 

Some things to remember:
- Yes we are implementing Contextual Retrieval as well (see [`ContextualVectorDB`](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb))
- Use `rich` for debugging messages
- Important to implement Prefect
- IT IS IMPORTANT TO STORE THE METADATA 
- allow verbosity levels for each function
- keep in mind that this will grow to be a complex system so the codebase needs to be designed in a modular manner. As an example, since we're planning on eventually having dozens of chunking strategies - we'll need to create a "chunking" directory, where each file contains a different chunking strategy.
- we'll have a .env file that defines - several things. Here is an example:
```
# Document Paths

HOME_DIRECTORY = "/home/blurry/"
MY_DOCS_PATH = "/home/blurry/Desktop/MIT-LLM/Docs/w6"
DEFAULT_DOCS_PATH = "home/blurry/Documents"

# Embedding

CHUNK_SIZE = 300
CHUNK_OVERLAP = 25

EMBEDDING_ALT = "/home/blurry/Desktop/ML/models/embedding/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL = "/home/blurry/Desktop/ML/models/embedding/multilingual-e5-large-instruct"

# Large Language Models

BASE_MODEL = "Qwen3.6-27B-UD-Q4_K_XL.gguf"
BASE_MODEL_API_URL = "http://localhost:8080/v1"

REASONING_MODEL = "Phi-4-reasoning-plus-UD-Q5_K_XL.gguf"
REASONING_MODEL_API_URL = "http://localhost:8080/v1"

TOOL_MODEL = "qwen2.5-coder-14b-instruct-q6_k.gguf"
TOOL_MODEL_API_URL = "http://localhost:8080/v1"

MAX_TOKENS = 8192
```




------------------------------------------------------


Features/Functionality that also needs to be added:
- Ability to use self hosted models (LLM + Embedding)
- Prefect to track every step of the RAG pipeline (Parsing, Chunking, Embedding, Store in DB)
- Evaluation system (https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide is a good place to look at)
- 'golden-docs' directory - explaining the system architecture

ONE THING THAT IS EXTREMELY ESSENTIAL - TESTS FOR EVERYTHING WE IMPLEMENT. ALSO LOGGING IS EXTREMELY IMPORTANT

After the initial version has been completed, 


Good RAG resources:
- READ THROUGH ALL pages here : https://jxnl.co/writing/2025/09/11/rag-series-index/#whats-next-for-rag

