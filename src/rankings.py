import bm25s


def get_bm25_ranking(query, retriever, k):
    bm25_results, _ = retriever.retrieve(bm25s.tokenize(query), k=k)
    return bm25_results[0].tolist()


def get_faiss_ranking(query, model, index, k):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    _, faiss_indices = index.search(query_embedding, k=k)
    return faiss_indices[0].tolist()


def get_rrf_ranking(faiss_ranking, bm25_ranking, k=60):
    scores = {}
    for rank, doc_idx in enumerate(faiss_ranking, start=1):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (k + rank)
    for rank, doc_idx in enumerate(bm25_ranking, start=1):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def rerank(cross_encoder, query, papers):
    # Create (query, document) pairs.
    pairs = [(query, p['title'] + ' ' + p['abstract']) for _, p in papers]
    ce_scores = cross_encoder.predict(pairs)
    return [
        idx for _, idx in sorted(
            zip(ce_scores, [idx for idx, _ in papers]),
            reverse=True
        )
    ]
