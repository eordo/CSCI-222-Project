# RAG for Domain-Specific QA 

This is my final project for CSCI 222 "Foundations of Large Language Models."

## Setup

### Environment

This project was developed using a GPU on the HUIT Open OnDemand cluster with CUDA 12.9.
Note that it uses the CPU version of FAISS; I found that GPU acceleration is not needed for vector indexing a corpus at the size of that in this project, but the GPU is especially helpful for LLM generation.

The cleanest way to reproduce the workflow and results is in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Secrets

The project dataset consists of paper abstracts and metadata pulled from [OpenAlex](https://openalex.org/).
You can get a free API key [here](https://openalex.org/settings/api).

Your API key should go in a top-level `.env` file like so:

```bash
# .env
export API_KEY="your_api_key_here" 
```

Summarization and relevancy scoring are performed by an LLM, specifically [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
This requires access to the repository with a Hugging Face access token, so request access and a token if you do not have one already and save it in `.env`:

```bash
# .env
export HF_TOKEN="your_token_here"
```

## Results

The results reported in the paper (relevance score tables and UMAP visualization) are included in the `results` directory and can be reproduced by running `main.py`.
