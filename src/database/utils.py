def prepend_metadata(metadata: dict) -> str:
    parts = []
    for k in ['filename', 'ref', 'page_number', 'figure_number', 'doc_id']:
        if k in metadata:
            parts.append(f"{k}: {metadata[k]}")
    return "[" + " | ".join(parts) + "]"
