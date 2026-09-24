import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import app_services
import support_store

BUNDLE = {"question": "Как изменить категорию?", "seconds": 1.2,
          "result": {"status": "unknown", "answer": "Недостаточно информации.", "sources": []},
          "hits": [{"chunk_id": "doc_01_001", "title": "Инструкция", "text": "Текст инструкции",
                    "source_url": "https://example.org/help", "score": 0.8}]}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.db = patch.object(support_store, 'DB_PATH', Path(self.temp.name) / 'support.sqlite3')
        self.db.start()

    def tearDown(self):
        self.db.stop()
        self.temp.cleanup()

    def test_empty_question_does_not_run_inference(self):
        with patch.object(app_services, 'ask') as ask:
            app = AppTest.from_file(str(ROOT / 'app.py')).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            ask.assert_not_called()

    def test_question_handoff_and_close(self):
        with patch.object(app_services, 'ask', return_value=BUNDLE) as ask:
            app = AppTest.from_file(str(ROOT / 'app.py')).run()
            app.text_area[0].input(BUNDLE['question'])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            ask.assert_called_once_with(BUNDLE['question'])
            next(b for b in app.button if b.label == 'Передать оператору').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.success)
            rows = support_store.list_tickets()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0]['evidence_json']), BUNDLE['hits'])
            app.sidebar.radio[0].set_value('Обращения оператору').run()
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == 'Закрыть обращение').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(support_store.list_tickets()[0]['status'], 'closed')
            self.assertEqual(ask.call_count, 1)

    def test_source_answer_is_displayed(self):
        result = {"status": "answer", "answer": "Ответ по документу. [1]", "claims": [],
                  "sources": [{"number": 1, "title": "Документ", "url": "https://example.org/help"}]}
        with patch.object(app_services, 'ask', return_value={**BUNDLE, 'result': result}):
            app = AppTest.from_file(str(ROOT / 'app.py')).run()
            app.text_area[0].input('Вопрос')
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Ответ по документу' in m.value for m in app.markdown))

    def test_database_idempotency_and_quote_characters(self):
        question = "Товар 'special'; DROP TABLE tickets; --"
        first = support_store.create_ticket('request-1', question, BUNDLE['result'], BUNDLE['hits'])
        second = support_store.create_ticket('request-1', question, BUNDLE['result'], BUNDLE['hits'])
        self.assertEqual(first, second)
        self.assertEqual(len(support_store.list_tickets()), 1)
        self.assertEqual(support_store.list_tickets()[0]['question'], question)


if __name__ == '__main__':
    unittest.main()
