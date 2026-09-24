"""Frozen local evaluation: measure retrieval, response status and latency.

Content correctness must be reviewed against the saved evidence and rubric.
Never interpret status agreement as answer accuracy or support deflection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import statistics
import sys
import time
from urllib.parse import urlparse

from answer_question import ROOT, Settings, ChatClient, answer
from search_docs import MODEL, SentenceTransformer, load_index, search, read_json
from retrieval_context import expand_context


class RecordingClient(ChatClient):
    def __init__(self, settings):
        super().__init__(settings)
        self.outputs = []

    def complete(self, system, payload, verify=False):
        result = super().complete(system, payload, verify=verify)
        self.outputs.append({"stage": "relevance" if verify == "relevance" else "verify" if verify else "generate", "output": result})
        return result


def summarize(rows):
    positives = [r for r in rows if r['expected_status'] == 'answer']
    negatives = [r for r in rows if r['expected_status'] == 'unknown']
    return {
        'completed': len(rows),
        'status_matches': sum(r['result']['status'] == r['expected_status'] for r in rows),
        'answerable_count': len(positives),
        'answered_answerable': sum(r['result']['status'] == 'answer' for r in positives),
        'retrieval_hit_at_5': sum(r['retrieval_hit'] for r in positives),
        'refusal_expected_count': len(negatives),
        'refusals_on_negative': sum(r['result']['status'] == 'unknown' for r in negatives),
        'errors': sum(r['result']['status'] == 'error' for r in rows),
        'median_seconds': round(statistics.median(r['seconds'] for r in rows), 2) if rows else None,
        'max_seconds': max((r['seconds'] for r in rows), default=None),
    }


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cases', type=str, default='data/evaluation_cases.json')
    args = parser.parse_args()
    settings = Settings.from_env()
    if urlparse(settings.base_url).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('Evaluation requires a local model')
    cases_path = ROOT / args.cases
    cases = read_json(cases_path)
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        cases = cases[:args.limit]
    destination = ROOT / 'data/evaluations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination.mkdir(parents=True)
    index, chunks = load_index()
    model = SentenceTransformer(MODEL, device='cpu', local_files_only=True)
    client = RecordingClient(settings)
    report = {
        'started_at': datetime.now(timezone.utc).isoformat(), 'model': settings.model,
        'cases_sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        'answer_code_sha256': hashlib.sha256((ROOT / 'scripts/answer_question.py').read_bytes()).hexdigest(),
        'pipeline_sha256': {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() for name in ['search_docs.py', 'retrieval_context.py', 'question_scope.py', 'literal_evidence.py']},
        'index_metadata': read_json(ROOT / 'data/index/metadata.json'),
        'planned_cases': len(cases), 'note': 'Synthetic author-reviewed set; not independent real support traffic.',
        'cases': [],
    }
    print(f'Report: {destination}', flush=True)
    for case in cases:
        started = time.perf_counter()
        client.outputs = []
        direct_hits = search(model, index, chunks, case['question'], 5)
        hits = expand_context(direct_hits, chunks)
        result = answer(case['question'], hits, client, settings.min_score)
        row = {**case, 'result': result, 'retrieved': hits, 'model_outputs': client.outputs,
               'retrieval_hit': any(h['doc_id'] == case.get('expected_doc') for h in direct_hits),
               'seconds': round(time.perf_counter() - started, 2), 'manual_review': None}
        report['cases'].append(row)
        report['summary'] = summarize(report['cases'])
        (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{case['id']}: {result['status']} | {row['seconds']}s | {result['answer']}", flush=True)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
