import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from answer_question import answer, ChatClient, ModelError, Settings

HIT = {"chunk_id": "doc_09_001", "title": "Категория", "section": "Изменение",
       "text": "Выберите новое значение в поле Категория и нажмите Сохранить.",
       "source_url": "https://example.org/source", "score": 0.9}
DRAFT = {"status": "answer", "claims": [{"text": "Измените поле Категория и сохраните карточку.",
          "evidence": [{"chunk_id": HIT["chunk_id"], "excerpt_id": "p1"}]}]}
VERDICT = {"answers_question": True, "checks": [{"claim_index": 0, "explanation": "Шаги подтверждены источником",
                                                "qualifiers_preserved": True, "supported": True}]}


class FakeClient:
    def __init__(self, *outputs, relevance=True):
        self.relevance = relevance
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, *args, **kwargs):
        if kwargs.get("verify") == "relevance":
            self.calls += 1
            return {"analysis": "Сопоставление с вопросом", "answers_question": self.relevance}
        output = self.outputs[self.calls]
        self.calls += 1
        if isinstance(output, Exception):
            raise output
        return copy.deepcopy(output)


class AnswerTests(unittest.TestCase):
    def test_supported_but_irrelevant_answer_is_not_released(self):
        result = answer('Вопрос', [HIT], FakeClient(DRAFT, VERDICT, relevance=False))
        self.assertEqual(result['reason'], 'incomplete_answer')
        self.assertEqual(result['sources'], [])

    def test_factual_check_is_separate_from_user_question(self):
        calls = []
        class CaptureClient(FakeClient):
            def complete(self, system, payload, verify=False):
                calls.append((verify, payload))
                return super().complete(system, payload, verify=verify)
        answer('Можно ли изменить категорию?', [HIT], CaptureClient(DRAFT, VERDICT))
        self.assertNotIn('question', calls[1][1])
        self.assertEqual(calls[2][0], 'relevance')
        self.assertEqual(calls[2][1]['question'], 'Можно ли изменить категорию?')

    def test_explicit_exception_cannot_be_overruled_by_verifier(self):
        hit = {**HIT, "text": "Посторонние логотипы запрещены (логотип бренда можно)."}
        draft = copy.deepcopy(DRAFT)
        draft['claims'][0]['text'] = 'Любые логотипы запрещены.'
        client = FakeClient(draft, VERDICT)
        result = answer('Вопрос', [hit], client)
        self.assertEqual(result['status'], 'answer')
        self.assertNotIn('Любые логотипы запрещены', result['answer'])
        self.assertIn(hit['text'], result['answer'])
        self.assertEqual(client.calls, 3)

    def test_preserved_explicit_exception_can_be_verified(self):
        hit = {**HIT, "text": "Посторонние логотипы запрещены (логотип бренда можно)."}
        draft = copy.deepcopy(DRAFT)
        draft['claims'][0]['text'] = hit['text']
        self.assertEqual(answer('Вопрос', [hit], FakeClient(draft, VERDICT))['status'], 'answer')

    def test_dropped_exception_claim_is_removed_and_answer_rechecked(self):
        draft = copy.deepcopy(DRAFT)
        draft['claims'].append({**copy.deepcopy(DRAFT['claims'][0]), 'text': 'Всем всё запрещено.'})
        verdict = copy.deepcopy(VERDICT)
        verdict['checks'].append({'claim_index': 1, 'explanation': 'Потеряно исключение',
                                  'supported': False, 'qualifiers_preserved': False})
        client = FakeClient(draft, verdict, VERDICT)
        result = answer('Вопрос', [HIT], client)
        self.assertEqual(result['status'], 'answer')
        self.assertNotIn('Всем', result['answer'])
        self.assertEqual(client.calls, 4)

    def test_pruning_essential_claim_still_refuses(self):
        draft = copy.deepcopy(DRAFT)
        draft['claims'].append(copy.deepcopy(DRAFT['claims'][0]))
        verdict = copy.deepcopy(VERDICT)
        verdict['checks'].append({'claim_index': 1, 'explanation': 'Нет подтверждения',
                                  'supported': False, 'qualifiers_preserved': True})
        final = {**copy.deepcopy(VERDICT), 'answers_question': False}
        self.assertEqual(answer('Вопрос', [HIT], FakeClient(draft, verdict, final, relevance=False))['status'], 'unknown')

    def test_irrelevant_claim_can_be_pruned_only_with_fresh_completeness_check(self):
        draft = copy.deepcopy(DRAFT)
        draft['claims'].append({**copy.deepcopy(DRAFT['claims'][0]), 'text': 'Посторонняя тема.'})
        verdict = copy.deepcopy(VERDICT)
        verdict['answers_question'] = False
        verdict['checks'].append({'claim_index': 1, 'explanation': 'Не относится к вопросу',
                                  'supported': False, 'qualifiers_preserved': False})
        client = FakeClient(draft, verdict, VERDICT)
        result = answer('Вопрос', [HIT], client)
        self.assertEqual(result['status'], 'answer')
        self.assertNotIn('Посторонняя', result['answer'])
        self.assertEqual(client.calls, 4)

    def test_local_configuration_does_not_require_key(self):
        env = {"LLM_BASE_URL": "http://127.0.0.1:11434/v1", "LLM_MODEL": "rag-support-qwen3"}
        with patch.dict('os.environ', env, clear=True), patch('answer_question.load_dotenv'):
            settings = Settings.from_env()
        self.assertEqual(settings.api_key, '')
        self.assertEqual(settings.timeout, 180)

    def test_remote_http_and_missing_remote_key_are_rejected(self):
        for url in ['http://remote.example/v1', 'https://remote.example/v1', 'http://localhost.remote.example/v1']:
            with self.subTest(url=url), patch.dict('os.environ', {"LLM_BASE_URL": url, "LLM_MODEL": "m"}, clear=True), patch('answer_question.load_dotenv'):
                with self.assertRaises(ValueError):
                    Settings.from_env()

    def test_supported_answer_has_local_source(self):
        result = answer("Как изменить категорию?", [HIT], FakeClient(DRAFT, VERDICT))
        self.assertEqual(result["status"], "answer")
        self.assertEqual(result["sources"][0]["url"], HIT["source_url"])
        self.assertIn("[1]", result["answer"])
        self.assertEqual(result["claims"][0]["evidence"][0]["quote"], HIT["text"])

    def test_empty_retrieval_does_not_call_model(self):
        client = FakeClient()
        self.assertEqual(answer("Вопрос", [], client)["status"], "unknown")
        self.assertEqual(client.calls, 0)

    def test_threshold_rejects_before_api(self):
        client = FakeClient()
        self.assertEqual(answer("Вопрос", [HIT], client, 0.95)["reason"], "below_search_threshold")
        self.assertEqual(client.calls, 0)

    def test_unknown_never_displays_model_claims(self):
        result = answer("Баланс аккаунта?", [HIT], FakeClient({"status": "unknown", "claims": DRAFT["claims"]}))
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["sources"], [])

    def test_fabricated_source_or_quote_is_rejected(self):
        for change in [{"chunk_id": "invented"}, {"excerpt_id": "p999"}, {"quote": "Выдуманная цитата"}]:
            with self.subTest(change=change):
                draft = copy.deepcopy(DRAFT)
                draft["claims"][0]["evidence"][0].update(change)
                client = FakeClient(draft)
                self.assertEqual(answer("Вопрос", [HIT], client)["reason"], "invalid_evidence")
                self.assertEqual(client.calls, 1)

    def test_real_quote_does_not_make_unsupported_claim_valid(self):
        draft = copy.deepcopy(DRAFT)
        draft["claims"][0]["text"] = "Смена категории гарантирует рост продаж на 50%."
        verdict = {"answers_question": True, "checks": [{"claim_index": 0, "supported": False}]}
        result = answer("Вопрос", [HIT], FakeClient(draft, verdict))
        self.assertEqual(result["reason"], "verification_failed")
        self.assertNotIn("50%", result["answer"])

    def test_incomplete_or_malformed_checks_are_rejected(self):
        for verdict in [{"answers_question": True, "checks": []},
                        {"answers_question": True, "checks": [{"claim_index": True, "supported": True}]}]:
            with self.subTest(verdict=verdict):
                self.assertEqual(answer("Вопрос", [HIT], FakeClient(DRAFT, verdict))["status"], "unknown")

    def test_verifier_outage_never_releases_draft(self):
        result = answer("Вопрос", [HIT], FakeClient(DRAFT, ModelError("Сеть недоступна")))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["sources"], [])

    def test_arbitrary_draft_shapes_fail_closed(self):
        for claims in [None, {}, [None], [{"text": "Ответ", "evidence": None}],
                       [{"text": "Ответ", "evidence": [{"chunk_id": [], "quote": "текст"}]}]]:
            with self.subTest(claims=claims):
                result = answer("Вопрос", [HIT], FakeClient({"status": "answer", "claims": claims}))
                self.assertEqual(result["status"], "unknown")

    def test_http_request_and_json_response(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://api.example.org/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-secret")
            self.assertEqual(request.headers["OpenAI-Project"], "test-folder")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "verifier")
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertEqual(body["response_format"]["json_schema"]["name"], "verify_answer")
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps(VERDICT)}}]})
        settings = Settings("https://api.example.org/v1", "test-secret", "generator", "verifier")
        settings.project = "test-folder"
        client = ChatClient(settings, httpx.MockTransport(handler))
        self.assertEqual(client.complete("JSON", {}, verify=True), VERDICT)
        self.assertNotIn("test-secret", repr(settings))

    def test_http_failure_truncation_and_bad_json(self):
        responses = [httpx.Response(401, text="secret response body"),
                     httpx.Response(200, json={"choices": [{"finish_reason": "length"}]}),
                     httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]})]
        for response in responses:
            with self.subTest(response=response):
                client = ChatClient(Settings("https://api.example.org/v1", "secret", "model"),
                                    httpx.MockTransport(lambda request: response))
                with self.assertRaises(ModelError) as error:
                    client.complete("JSON", {})
                self.assertNotIn("secret", str(error.exception))


if __name__ == "__main__":
    unittest.main()
