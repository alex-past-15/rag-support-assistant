import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from answer_question import answer, source_excerpts, preserves_explicit_exceptions
from question_scope import needs_account_access
from retrieval_context import expand_context
from literal_evidence import literal_claim


class NeverClient:
    def complete(self, *args, **kwargs):
        raise AssertionError('Private account lookup must not call the LLM')


class GroundingTests(unittest.TestCase):
    def test_literal_fallback_keeps_list_intro_and_only_selected_bullet(self):
        excerpts = {'a': source_excerpts('Нельзя использовать:\n\n* низкое качество;\n* логотип магазина (логотип бренда можно).')}
        claim = {'text': 'Все логотипы запрещены.', 'evidence': [{'chunk_id': 'a', 'excerpt_id': 'p3', 'quote': excerpts['a']['p3']}]}
        result = literal_claim(claim, excerpts)
        self.assertIn('Нельзя использовать:', result['text'])
        self.assertIn('логотип бренда можно', result['text'])
        self.assertNotIn('низкое качество', result['text'])
        self.assertNotIn('Все логотипы', result['text'])
        self.assertEqual(len(result['evidence']), 2)

    def test_personal_diagnosis_is_routed_before_generation(self):
        for question in ['Почему именно мой товар 12345 скрыли вчера?',
                         'За что заблокировали наш товар?',
                         'Из-за чего отклонили товар AB-709?',
                         'Почему удалили карточку?',
                         'Какой сейчас баланс моего магазина?',
                         'Сколько заказов поступило в мой магазин сегодня?']:
            with self.subTest(question=question):
                result = answer(question, [], NeverClient())
                self.assertEqual(result['reason'], 'no_account_access')
                self.assertTrue(result['can_escalate'])

    def test_general_instructions_are_not_account_lookups(self):
        for question in ['Как изменить категорию моего товара?',
                         'Почему могут скрыть товар?',
                         'Какие типичные причины удаления товаров?',
                         'Фото 1200 на 1200 подойдёт?',
                         'После восстановления товар появится во всех магазинах?']:
            with self.subTest(question=question):
                self.assertFalse(needs_account_access(question))

    def test_list_exceptions_do_not_leak_between_items(self):
        text = 'Не подходят:\n\n* логотип магазина (логотип бренда можно);\n* посторонние люди (кроме пользователей товара).'
        excerpts = source_excerpts(text)
        self.assertEqual(len(excerpts), 3)
        for quote in excerpts.values():
            self.assertIn(quote, text)
        claim = {'text': 'Логотип бренда можно.', 'evidence': [{'quote': excerpts['p2']}]}
        self.assertTrue(preserves_explicit_exceptions(claim))
        claim['text'] = 'Любой логотип запрещён.'
        self.assertFalse(preserves_explicit_exceptions(claim))

    def test_parent_conditions_are_added_once_and_from_same_document(self):
        child = {'chunk_id': 'a2', 'doc_id': 'a', 'section': 'Возврат / В кабинете', 'text': 'Нажмите вернуть.', 'score': .8}
        parent = {'chunk_id': 'a1', 'doc_id': 'a', 'section': 'Возврат', 'text': 'Только если задана цена.'}
        other = {**parent, 'chunk_id': 'b1', 'doc_id': 'b'}
        rows = expand_context([child], [other, parent, child])
        self.assertEqual([r['chunk_id'] for r in rows], ['a2', 'a1'])
        self.assertEqual(rows[1]['context_of'], 'a2')
        self.assertEqual(len(expand_context(rows, [parent, child])), 2)
        self.assertEqual(len(expand_context([child], [parent], max_chars=1)), 1)


if __name__ == '__main__':
    unittest.main()
