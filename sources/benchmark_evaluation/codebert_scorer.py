"""
CodeBERT Scorer - Calculate similarity between generated and gold code.

Uses CodeBERT embeddings to compute token-level F1 score between
generated code and reference (gold) code.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def calculate_codebert_score(
    generated_code_path: Path,
    gold_code_path: Path,
) -> float:
    """
    Calculate CodeBERTScore between generated and gold code.
    Args:
        generated_code_path: Path to generated Python code
        gold_code_path: Path to gold (reference) Python code
    Returns:
        CodeBERT F1 score (0.0-1.0)
    """
    try:
        try:
            return _calculate_with_codebert(
                generated_code_path,
                gold_code_path
            )
        except ImportError:
            logger.warning("[CBS] transformers library not available")
            return 0.0
    except Exception as e:
        logger.error(f"[CBS] Error calculating CodeBERT score: {str(e)}")
        return 0.0

def _calculate_with_codebert(
    generated_code_path: Path,
    gold_code_path: Path
) -> float:
    """
    Calculate CodeBERTScore using CodeBERT model.
    """
    try:
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

        # Use tokenizer's encode method with proper truncation and padding
        # This handles max_length properly and avoids the warning
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
            # Get embeddings
            gen_outputs = model(**gen_encoding)
            gold_outputs = model(**gold_encoding)

            # Get last hidden states [batch_size, seq_len, hidden_size]
            gen_embeddings = gen_outputs.last_hidden_state[0]  # [seq_len, hidden_size]
            gold_embeddings = gold_outputs.last_hidden_state[0]

            # Get attention masks to ignore padding tokens
            gen_mask = gen_encoding['attention_mask'][0].bool()
            gold_mask = gold_encoding['attention_mask'][0].bool()

            # Filter out padding tokens
            gen_embeddings = gen_embeddings[gen_mask]
            gold_embeddings = gold_embeddings[gold_mask]

        # Calculate similarity matrix
        logger.info("[CBS] Calculating similarity...")
        # Normalize embeddings
        gen_normalized = F.normalize(gen_embeddings, p=2, dim=1)
        gold_normalized = F.normalize(gold_embeddings, p=2, dim=1)

        # Compute cosine similarity matrix
        similarity = torch.mm(gen_normalized, gold_normalized.t())

        # Calculate F1 score using greedy matching
        gen_to_gold_scores = similarity.max(dim=1)[0]  # Best match for each gen token
        gold_to_gen_scores = similarity.max(dim=0)[0]  # Best match for each gold token

        # Calculate precision and recall
        precision = gen_to_gold_scores.mean().item()
        recall = gold_to_gen_scores.mean().item()

        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        logger.info(f"[CBS] CodeBERT F1 score: {f1:.3f} (P={precision:.3f}, R={recall:.3f})")
        return float(f1)
    except Exception as e:
        logger.error(f"[CBS] Error in CodeBERT calculation: {str(e)}")
        raise


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
