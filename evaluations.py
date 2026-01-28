from rouge import Rouge
import numpy as np

rouge = Rouge()

def _to_id_list(token_ids):
    # Accept list/np/tensor; return plain python list of non-negative ints.
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu()
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, (list, tuple)):
        return [int(t) for t in token_ids if int(t) >= 0]
    return []


def rouge_score(hyp_ids, ref_ids, tokenizer):
    hyp_ids = _to_id_list(hyp_ids)
    ref_ids = _to_id_list(ref_ids)
    if not hyp_ids or not ref_ids:
        return 0.0
    hyps = [tokenizer.decode(hyp_ids, skip_special_tokens=True)]
    if len(hyps[0]) == 0:
        return 0.0
    refs = [tokenizer.decode(ref_ids, skip_special_tokens=True)]
    try:
        rouge_score = rouge.get_scores(hyps, refs)[0]['rouge-l']['f']
    except ValueError:
        return 0.0
    return rouge_score


def acc_score(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return np.sum(preds == labels) / float(len(labels))
