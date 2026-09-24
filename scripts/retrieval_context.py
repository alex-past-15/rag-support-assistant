"""Add introductory conditions of parent sections without rebuilding vectors."""


def expand_context(hits, chunks, max_added=4, max_chars=6000):
    result = [dict(h) for h in hits]
    seen = {h['chunk_id'] for h in hits}
    added_chars = 0
    for hit in hits:
        parts = hit['section'].split(' / ')
        parents = [' / '.join(parts[:i]) for i in range(len(parts) - 1, 0, -1)]
        for section in parents:
            for chunk in chunks:
                if chunk['doc_id'] != hit['doc_id'] or chunk['section'] != section or chunk['chunk_id'] in seen:
                    continue
                if len(result) - len(hits) >= max_added or added_chars + len(chunk['text']) > max_chars:
                    continue
                # The score belongs to the originating hit, not to a new vector search.
                result.append({**chunk, 'score': hit['score'], 'context_of': hit['chunk_id']})
                seen.add(chunk['chunk_id'])
                added_chars += len(chunk['text'])
    return result
