import re
from transformers import GenerationConfig


_generator = None
_score_cache = {}

_JUDGE_PROMPT = (
    "You are an economics researcher evaluating whether an academic paper is "
    "relevant to a search query.\n\n"
    "Query:\n"
    "{query}\n\n"
    "Paper title:\n"
    "{title}\n\n"
    "Abstract:\n"
    "{abstract}\n\n"
    "Assign exactly one relevance score:\n\n"
    "0 = Not relevant\n"
    "The paper does not meaningfully address the query, even if it shares "
    "keywords.\n\n"
    "1 = Tangentially related\n"
    "The paper belongs to a related literature or topic area, but does not "
    "directly address the core information need.\n\n"
    "2 = Relevant\n"
    "The paper substantially addresses the query, but misses one or more "
    "important aspects (e.g., population, method, geography, time period, or "
    "research focus.\n\n"
    "3 = Highly relevant\n"
    "The paper directly addresses the query and would likely be among the "
    "best papers to answer the information need.\n\n"
    "Instructions:\n"
    "- Use only the title and abstract.\n"
    "- Do not assume contributions unless explicitly stated.\n"
    "- Be conservative in assigning high scores.\n\n"
    "Respond in exactly this format:\n\n"
    "Reasoning: <one sentence>\n"
    "Score: <0, 1, 2, or 3>"
)
_SUMMARY_PROMPT = (
    "A researcher has asked: {query}\n\n"
    "Below are the most relevant papers retrieved from an economics "
    "literature corpus:\n\n"
    "{context}\n\n"
    "Synthesize the key findings of these papers in 2-3 paragraphs. "
    "Focus on:\n"
    "- The main empirical findings and theoretical contributions\n"
    "- Points of consensus and disagreement across papers\n"
    "- How the literature has evolved over time\n\n"
    "Cite papers by number, e.g., [1], [2]."
)


def init_generator(generator):
    global _generator
    generator.tokenizer.pad_token = generator.tokenizer.eos_token
    generator.tokenizer.padding_side = 'left'
    _generator = generator


def score(query, papers, batch_size=8):
    if _generator is None:
        raise RuntimeError("Run `init_generator` before calling `score`.")
    
    # Partition into cached and uncached.
    to_score = []
    for idx, p in papers:
        key = (query, idx)
        if key not in _score_cache:
            to_score.append((idx, p))
    
    if to_score:
        # Build batched messages list.
        all_messages = [
            [
                {'role': 'system',
                 'content': "You are an expert economics researcher."},
                {'role': 'user',
                 'content': _JUDGE_PROMPT.format(
                     query=query,
                     title=p['title'],
                     abstract=p['abstract'][:800]
                 )}
            ]
            for _, p in to_score
        ]
        gen_config = GenerationConfig(
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=_generator.tokenizer.eos_token_id
        )
        outputs = _generator(
            all_messages,
            generation_config=gen_config,
            batch_size=batch_size
        )
        for (idx, p), output in zip(to_score, outputs):
            response = output[0]['generated_text'][-1]['content'].strip()
            _score_cache[(query, idx)] = {
                'id': idx,
                'score': _parse_score(response),
                'reasoning': response
            }

    return [_score_cache[(query, idx)] for idx, _ in papers]


def summarize(query, papers, max_new_tokens=512):
    if _generator is None:
        raise RuntimeError("Call `init_generator` before calling `summarize`.")
    
    messages = [
        {'role': 'system',
         'content': "You are a research assistant for academic economics."},
        {'role': 'user',
         'content': _build_prompt(query, papers)}
    ]
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=_generator.tokenizer.eos_token_id
    )
    output = _generator(messages, generation_config=gen_config)

    return output[0]['generated_text'][-1]['content']


def _build_prompt(query, papers):
    context = '\n\n'.join(
        f"[{rank}] {paper['title']} ({paper['year']})\n"
        f"Abstract: {paper['abstract']}"
        for rank, (_, paper) in enumerate(papers, start=1)
    )
    return _SUMMARY_PROMPT.format(query=query, context=context)


def _parse_score(text):
    score_match = re.search(r"Score:\s*([0123])", text)
    if score_match:
        return int(score_match.group(1))
    digits = re.findall(r'\b([0123])\b', text)
    return int(digits[-1]) if digits else None
