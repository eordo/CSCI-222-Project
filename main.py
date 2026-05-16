# Imports can take a while, especially on the cluster.
print("Setting up environment - please wait...")

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import bm25s
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
import umap
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

from src.evaluation import init_generator, score, summarize
from src.rankings import (get_bm25_ranking, get_faiss_ranking,
                          get_rrf_ranking, rerank)


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
# Mount Google Drive if running in Colab.
# Edit this Path to be the notebook's location relative to MyDrive.
GDRIVE_PATH = Path('HES/CSCI_222/Project')
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    PROJECT_ROOT = Path('/content/drive/MyDrive') / GDRIVE_PATH
else:
    PROJECT_ROOT = Path('.')

os.chdir(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Use a GPU if available.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USING_GPU = DEVICE.type == 'cuda'
print(f'Device: {DEVICE}')
if USING_GPU:
    print(f"  GPU:  {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load the project .env.
assert load_dotenv(PROJECT_ROOT / '.env'), "Failed to load .env, check path."

# Load the OpenAlex API key.
BASE_URL = "https://api.openalex.org/works"
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise RuntimeError('OpenAlex API key not found in .env.')

# Load the Hugging Face token.
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise RuntimeError('Hugging Face token not found in .env.')
login(token=HF_TOKEN)

# ============================================================================
# DATA LOADING
# ============================================================================
papers_50k_json = DATA_DIR / 'papers_50k.json'
if papers_50k_json.exists():
    with open(papers_50k_json, 'r') as f:
        papers = json.load(f)
        print(f"Loaded {len(papers):,} papers from {papers_50k_json}.")
else:
    def reconstruct_abstract(inverted_index):
        positions = [
            (pos, word)
            for word, positions in inverted_index.items()
            for pos in positions
        ]
        return ' '.join(word for _, word in sorted(positions))

    filter_fields = [
        'primary_topic.field.id:20',
        'type:article',
        'has_abstract:true',
        'is_paratext:false', # Non-publication material.
        'is_retracted:false',
        'language:en'
    ]
    select_fields = [
        'id',
        'title',
        'authorships',
        'publication_year',
        'primary_location',
        'abstract_inverted_index',
        'primary_topic',
        'doi'
    ]
    params = {
        'filter': ','.join(filter_fields),
        'select': ','.join(select_fields),
        'per_page': 200,
        'cursor': '*',
        'api_key': API_KEY
    }

    PAPERS_LIMIT = 50_000
    papers = []
    while len(papers) < PAPERS_LIMIT:
        resp = requests.get(BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        for work in data['results']:
            abstract = reconstruct_abstract(work['abstract_inverted_index'])
            # Skip missing abstracts and missing titles.
            if not abstract or not work.get('title'):
                continue
            authors = [a['author']['display_name'] for a in work.get('authorships', [])]
            source = work.get('primary_location', {}).get('source', {})
            journal = source.get('display_name') if source is not None else None
            papers.append({
                'id': work['id'],
                'title': work.get('title'),
                'year': work.get('publication_year'),
                'authors': authors,
                'abstract': abstract,
                'journal': journal,
                'topic': work.get('primary_topic', {}).get('display_name'),
                'doi': work.get('doi')
            })
            if len(papers) == PAPERS_LIMIT:
                break

        print(f"Collected {len(papers)} papers...", end='\r', flush=True)
        next_cursor = data['meta'].get('next_cursor')
        if not next_cursor:
            break
        params['cursor'] = next_cursor
        time.sleep(0.5) # Polite backoff.

    with open(papers_50k_json, 'w') as f:
        json.dump(papers, f)
        print(f"Saved {PAPERS_LIMIT:,} papers to {papers_50k_json}.")

# ============================================================================
# DENSE EMBEDDINGS
# ============================================================================
model_name = 'allenai/specter2_base'
model = SentenceTransformer(model_name, device=DEVICE)

specter_embeddings_50k_npy = DATA_DIR / 'specter_embeddings_50k.npy'
if specter_embeddings_50k_npy.exists():
    embeddings = np.load(specter_embeddings_50k_npy)
    print(f"Loaded SPECTER embeddings from {specter_embeddings_50k_npy}.")
else:
    texts = [p['title'] + " [SEP] " + p['abstract'] for p in papers]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings = embeddings.astype('float32')
    np.save(specter_embeddings_50k_npy, embeddings)
    print(f"Saved SPECTER embeddings to {specter_embeddings_50k_npy}.")

# ============================================================================
# INDEXES
# ============================================================================
# FAISS index.
faiss_index_file = DATA_DIR / 'specter_index_50k.faiss'
if faiss_index_file.exists():
    index = faiss.read_index(str(faiss_index_file))
    print(f"Loaded FAISS index from {faiss_index_file}.")
else:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(
        index,
        str(DATA_DIR / 'specter_index_50k.faiss')
    )
    print(f"Saved FAISS index to {faiss_index_file}.")

# BM25 index.
bm25s_index_dir = DATA_DIR / 'bm25s_50k'
if bm25s_index_dir.exists():
    retriever = bm25s.BM25.load(bm25s_index_dir)
    print(f"Loaded retriever index from {bm25s_index_dir}.")
else:
    corpus = [p['title'] + " " + p['abstract'] for p in papers]
    corpus_tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(bm25s_index_dir)
    print(f"Saved retriever index to {bm25s_index_dir}.")

# ============================================================================
# MODELS
# ============================================================================
# Load MS Marco Cross-Encoder.
# This is for reranking.
ce_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
cross_encoder = CrossEncoder(ce_name, device=DEVICE)

# Load Llama 3.1.
# This is for abstract summarization and scoring.
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
llm_name = 'meta-llama/Llama-3.1-8B-Instruct'
llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    quantization_config=bnb_config,
    torch_dtype='auto',
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(
    llm_name,
    clean_up_tokenization_spaces=False
)
generator = pipeline(
    'text-generation',
    model=llm,
    tokenizer=tokenizer,
)
# Initialize the module-level generator.
init_generator(generator)

# ============================================================================
# EVALUATION
# ============================================================================
scores_csv = RESULTS_DIR / 'scores.csv'
if scores_csv.exists():
    records_df = pd.read_csv(scores_csv)
    print(f"Loaded relevance scores from {scores_csv}.")
else:
    # Load the test queries.
    df = pd.read_csv('queries.csv')

    k = 50      # Depth for FAISS/BM25 search.
    top_k = 10  # Return top 10 papers.
    records = []
    print(f"Retrieving and scoring abstracts for {len(df)} queries.")
    for i, (query, category) in enumerate(df.itertuples(index=False, name=None)):
        print(f"[{i+1}] {query}")
        # FAISS.
        faiss_ranking = get_faiss_ranking(query, model, index, k=k)
        faiss_top_k   = [(idx, papers[idx]) for idx in faiss_ranking[:top_k]]
        # BM25.
        bm25_ranking = get_bm25_ranking(query, retriever, k=k)
        bm25_top_k   = [(idx, papers[idx]) for idx in bm25_ranking[:top_k]]
        # RRF.
        rrf_ranking = get_rrf_ranking(faiss_ranking, bm25_ranking)
        rrf_top_k   = [(idx, papers[idx]) for idx in rrf_ranking[:top_k]]
        # Reranked.
        reranking = rerank(
            cross_encoder,
            query,
            [(i, papers[i]) for i in rrf_ranking]
        )
        reranked_top_k = [(idx, papers[idx]) for idx in reranking[:top_k]]

        # Score the relevance of each retrieved abstract.
        all_papers = faiss_top_k + bm25_top_k + rrf_top_k + reranked_top_k
        all_scores = score(query, all_papers, batch_size=4)
        scores = {
            'faiss':    all_scores[0:top_k],
            'bm25':     all_scores[top_k:2*top_k],
            'rrf':      all_scores[2*top_k:3*top_k],
            'reranked': all_scores[3*top_k:4*top_k]
        }
        for method, method_scores in scores.items():
            for rank, r in enumerate(method_scores, start=1):
                records.append({
                    'query': query,
                    'category': category,
                    'method': method,
                    'rank': rank,
                    'score': r['score'] if r['score'] is not None else None
                })

        torch.cuda.empty_cache()

    records_df = pd.DataFrame(records)
    records_df.to_csv(scores_csv, index=False)
    print(f"Saved relevance scores @{top_k} to {scores_csv}.")
