"""
CodeBERT Scorer - Calculate similarity between generated and gold code.

Uses CodeBERT embeddings to compute token-level F1 score between
generated code and reference (gold) code.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_codebert_score(
    generated_code_path: Path,
    gold_code_path: Path,
) -> float:
    """
    Calculate CodeBERTScore between generated and gold code.

    Raises on failure (missing transformers/torch, unreadable files, model
    errors) so the caller can record WHY CBS is unavailable instead of silently
    reporting a plausible 0.0. See CapsuleEvaluator.calculate_codebert_score,
    which owns the fallback-and-log policy.

    Args:
        generated_code_path: Path to generated Python code
        gold_code_path: Path to gold (reference) Python code
    Returns:
        CodeBERT F1 score (0.0-1.0)
    """
    return _calculate_with_codebert(generated_code_path, gold_code_path)

def _calculate_with_codebert(
    generated_code_path: Path,
    gold_code_path: Path
) -> float:
    """
    Calculate CodeBERTScore using the CodeBERT model (greedy token-embedding F1).
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel

    logger.info("[CBS] Loading CodeBERT model...")

    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    with open(generated_code_path, encoding='utf-8') as f:
        generated_code = f.read()

    with open(gold_code_path, encoding='utf-8') as f:
        gold_code = f.read()

    logger.info("[CBS] Tokenizing and encoding code...")

    # Truncate/pad to the model's max length; padding is masked out below.
    max_length = 512

    gen_encoding = tokenizer(
        generated_code,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    gold_encoding = tokenizer(
        gold_code,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    logger.info("[CBS] Computing embeddings...")
    with torch.no_grad():
        gen_outputs = model(**gen_encoding)
        gold_outputs = model(**gold_encoding)

        # last_hidden_state: [batch_size, seq_len, hidden_size]
        gen_embeddings = gen_outputs.last_hidden_state[0]
        gold_embeddings = gold_outputs.last_hidden_state[0]

        # Drop padding tokens via the attention mask
        gen_mask = gen_encoding['attention_mask'][0].bool()
        gold_mask = gold_encoding['attention_mask'][0].bool()
        gen_embeddings = gen_embeddings[gen_mask]
        gold_embeddings = gold_embeddings[gold_mask]

    logger.info("[CBS] Calculating similarity...")
    gen_normalized = F.normalize(gen_embeddings, p=2, dim=1)
    gold_normalized = F.normalize(gold_embeddings, p=2, dim=1)

    # Cosine similarity matrix, then greedy-match F1
    similarity = torch.mm(gen_normalized, gold_normalized.t())
    gen_to_gold_scores = similarity.max(dim=1)[0]  # best gold match per gen token
    gold_to_gen_scores = similarity.max(dim=0)[0]  # best gen match per gold token

    precision = gen_to_gold_scores.mean().item()
    recall = gold_to_gen_scores.mean().item()

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    logger.info(f"[CBS] CodeBERT F1 score: {f1:.3f} (P={precision:.3f}, R={recall:.3f})")
    return float(f1)


def preload_codebert_model() -> tuple | None:
    """
    Preload CodeBERT model to cache for faster subsequent scoring.
    Returns:
        (tokenizer, model) tuple or None if loading fails
    """
    try:
        from transformers import AutoTokenizer, AutoModel

        logger.info("[CBS] Preloading CodeBERT model...")
        model_name = "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        logger.info("[CBS] CodeBERT model loaded successfully")
        return (tokenizer, model)
    except ImportError:
        logger.warning("[CBS] transformers library not available for preloading")
        return None
    except Exception as e:
        logger.error(f"[CBS] Error preloading CodeBERT model: {str(e)}")
        return None


if __name__ == "__main__":
    # Smoke check: scoring identical files should give a high F1; when
    # transformers/torch is absent the scorer must RAISE (caller owns fallback).
    import tempfile
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as _d:
        _p = Path(_d) / "prog.py"
        _p.write_text("import numpy as np\nprint(np.mean([1, 2, 3]))\n")
        try:
            _score = calculate_codebert_score(_p, _p)
            assert 0.0 <= _score <= 1.0, _score
            print(f"codebert_scorer smoke check passed (self-similarity F1={_score:.3f})")
        except (ImportError, ModuleNotFoundError):
            print("codebert_scorer smoke check passed (transformers absent → raised as expected)")
