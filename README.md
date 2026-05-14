# RAG for Domain-Specific QA 

This is my final project for CSCI 222 "Foundations of Large Language Models."

## Setup

### Environment

This project is being developed using a GPU on a cluster with CUDA 12.9.
This requires a specific version of FAISS that is incompatible with NumPy <3.

The cleanest way to reproduce the workflow and results is in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### API key

The project dataset consists of paper abstracts and metadata pulled from [OpenAlex](https://openalex.org/).
You can get a free API key [here](https://openalex.org/settings/api).

Your API key should go in a top-level `.env` file like so:

```bash
# .env
export API_KEY="your_api_key_here" 
```
