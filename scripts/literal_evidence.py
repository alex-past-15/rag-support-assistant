"""Replace a risky paraphrase with exact, labelled source excerpts."""
import re


def literal_claim(claim, excerpts):
    refs, seen = [], set()
    for ref in claim['evidence']:
        parts = list(excerpts[ref['chunk_id']].items())
        position = next(i for i, (key, _) in enumerate(parts) if key == ref['excerpt_id'])
        # A bullet can complete an introductory prohibition or condition. Preserve
        # the list lead-in as another exact quote, not a fabricated combined quote.
        if re.match(r'\s*[*-] ', ref['quote']):
            for key, text in reversed(parts[:position]):
                if re.match(r'\s*[*-] ', text):
                    continue
                if text.rstrip().endswith(':'):
                    head = {'chunk_id': ref['chunk_id'], 'excerpt_id': key, 'quote': text}
                    if (ref['chunk_id'], key) not in seen:
                        refs.append(head)
                        seen.add((ref['chunk_id'], key))
                break
        identity = (ref['chunk_id'], ref['excerpt_id'])
        if identity not in seen:
            refs.append(ref)
            seen.add(identity)
    text = 'В справке указано:\n\n' + '\n\n'.join('«' + r['quote'] + '»' for r in refs)
    if len(text) > 1500 or len(refs) > 5:
        return None
    return {'text': text, 'evidence': refs}
