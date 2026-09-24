"""Conservative Russian rules for requests that require private account facts.

These are explicit routing rules, not a universal intent classifier.
"""
import re


def needs_account_access(question):
    q = question.casefold().replace('ё', 'е')
    personal = bool(re.search(r'\b(?:мой|моя|мое|мои|моего|моей|моем|моих|наш\w*|у меня|у нас)\b', q))
    identifier = bool(re.search(r'\b(?:товар\w*|заказ\w*|sku|артикул\w*)\s*(?:№|номер|с номером)?\s*[a-zа-я_-]*\d[\w-]*', q))
    diagnosis = bool(re.search(r'почему|причин\w*|за что|из-за чего', q))
    direct_lookup = bool(re.search(r'баланс|сколько.*(?:заказ|продаж|денег)|(?:какой|узна[йт]).*статус', q))
    concrete_event = bool(re.search(r'\b(?:заблокировали|скрыли|отклонили|удалили)\b', q))
    generic = bool(re.search(r'\b(?:могут|может|обычно|в целом|возможные|типичные)\b', q))
    return ((personal or identifier) and (diagnosis or direct_lookup)) or (diagnosis and concrete_event and not generic)
