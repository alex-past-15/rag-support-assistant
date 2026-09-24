"""Local inference shared by UI sessions. Models load only on the first question."""
from functools import lru_cache
from threading import Lock
from time import perf_counter
from urllib.parse import urlparse

from answer_question import Settings, ChatClient, answer

_inference_lock = Lock()


@lru_cache(maxsize=1)
def embedding_model():
    from search_docs import MODEL, SentenceTransformer
    return SentenceTransformer(MODEL, device="cpu", local_files_only=True)


def ask(question):
    question = question.strip()
    if not question:
        raise ValueError("Введите вопрос.")
    settings = Settings.from_env()
    if urlparse(settings.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Этот интерфейс настроен для локальной модели. Укажите адрес Ollama в .env.")
    from search_docs import load_index, search
    started = perf_counter()
    with _inference_lock:
        index, chunks = load_index()
        from retrieval_context import expand_context
        hits = expand_context(search(embedding_model(), index, chunks, question, 5), chunks)
        result = answer(question, hits, ChatClient(settings), settings.min_score)
    return {"question": question, "result": result,
            "hits": [{k: h[k] for k in ("chunk_id", "title", "section", "text", "source_url", "score")} for h in hits],
            "seconds": round(perf_counter() - started, 1)}
