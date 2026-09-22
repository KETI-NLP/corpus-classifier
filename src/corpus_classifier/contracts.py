"""Frozen input formatting, taxonomy decoding and document-state layout."""
import json

STATE_LABELS = {
    "document_integrity": ["boilerplate", "coherent", "fragment", "stitched"],
    "topic_structure": ["indeterminate", "multi_topic", "single_topic"],
    "labelability": ["insufficient_evidence", "labelable"],
}

def document_text(row):
    # Never pass upstream labels, roots, path locators or selection metadata.
    metadata = row.get("metadata_compact") or ""
    return f"[DOCUMENT METADATA]\n{metadata}\n\n[DOCUMENT]\n{row['text_window']}"

def clean_metadata(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return ""
    if not isinstance(value, dict):
        return ""
    allowed = {"title", "language", "date", "publisher", "format_family"}
    output = {key: value[key] for key in allowed if value.get(key)}
    nested = value.get("metadata_subset")
    if isinstance(nested, dict):
        output.update({key: nested[key] for key in allowed if nested.get(key)})
    return json.dumps(output, ensure_ascii=False, sort_keys=True)[:1200]

def nonredundant(codes):
    selected = []
    for code in codes:
        if not any(code.startswith(prior + ".") or prior.startswith(code + ".") for prior in selected):
            selected.append(code)
    return selected

def predict_sets(root_scores, exact_scores, contract, root_threshold, exact_threshold, roots=None):
    selected_roots = roots if roots is not None else [r for r, p in zip(contract["roots"], root_scores) if p >= root_threshold]
    predictions = []
    for root in selected_roots:
        indices = [i for i, j in enumerate(contract["exact_root_indices"]) if contract["roots"][j] == root]
        indices.sort(key=lambda i: (-exact_scores[i], contract["exact"][i]))
        predictions += nonredundant([contract["exact"][i] for i in indices if exact_scores[i] >= exact_threshold])
    return selected_roots, sorted(predictions)
