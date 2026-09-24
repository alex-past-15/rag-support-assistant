"""Streamlit UI. Run: python -m streamlit run rag-support/app.py"""
import json
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import app_services
import support_store

st.set_page_config(page_title="Помощник продавца", page_icon="💬", layout="centered")
st.sidebar.title("Помощник продавца")
st.sidebar.caption("По справке Яндекс Маркета")
page = st.sidebar.radio("Раздел", ["Задать вопрос", "Обращения оператору"])
st.sidebar.divider()
st.sidebar.caption("Учебный проект. Модель работает на этом компьютере. Не официальный сервис Яндекса.")

if page == "Задать вопрос":
    st.title("Разберёмся с карточкой товара")
    st.write("Спросите о загрузке товаров, фотографиях, характеристиках или статусах. Ответ сопровождается источниками.")
    with st.form("question_form"):
        question = st.text_area("Ваш вопрос", placeholder="Например: можно ли добавить логотип бренда на фото товара?",
                                max_chars=1500, height=100, key="question")
        submitted = st.form_submit_button("Найти ответ", type="primary")
    st.caption("Ответ обычно занимает несколько десятков секунд. Первый запуск может быть дольше.")
    if submitted:
        st.session_state.pop("response", None)
        st.session_state.pop("ticket_id", None)
        if not question.strip():
            st.warning("Введите вопрос перед отправкой.")
        else:
            try:
                with st.spinner("Ищу инструкцию и проверяю ответ…"):
                    st.session_state.response = app_services.ask(question)
                st.session_state.request_key = str(uuid4())
            except ValueError:
                st.error("Не удалось обработать вопрос. Попробуйте сократить его; если ошибка повторится, проверьте настройки проекта.")
            except (OSError, RuntimeError):
                st.error("Не удалось загрузить базу знаний или модель. Проверьте, что установка завершена и индекс создан.")
    if "response" in st.session_state:
        bundle = st.session_state.response
        result = bundle["result"]
        st.subheader("Ответ")
        st.caption(bundle["question"])
        if result["status"] == "answer":
            st.markdown(result["answer"])
        elif result["status"] == "unknown":
            st.warning(result["answer"])
        else:
            st.error(result["answer"])
            st.caption("Проверьте, что Ollama запущен, и повторите вопрос.")
        if result["sources"]:
            st.subheader("Источники")
            for source in result["sources"]:
                st.link_button(f"[{source['number']}] {source['title']}", source["url"])
            with st.expander("Показать подтверждающие цитаты"):
                for claim in result.get("claims", []):
                    st.write(claim["text"])
                    for ref in claim["evidence"]:
                        st.text(ref["quote"])
        st.caption(f"Время обработки: {bundle['seconds']} с. Важные условия можно проверить по источнику.")
        st.divider()
        st.caption("Передача оператору сохраняет вопрос в локальную очередь этого проекта. В поддержку Яндекса ничего не отправляется.")
        if st.button("Передать оператору", disabled="ticket_id" in st.session_state):
            try:
                st.session_state.ticket_id = support_store.create_ticket(
                    st.session_state.request_key, bundle["question"], result, bundle["hits"])
                st.rerun()
            except sqlite3.Error:
                st.error("Не удалось сохранить обращение. Попробуйте ещё раз.")
        if "ticket_id" in st.session_state:
            st.success(f"Обращение №{st.session_state.ticket_id} сохранено. Оно доступно в разделе «Обращения оператору».")
else:
    st.title("Обращения оператору")
    st.caption("Локальная демонстрационная очередь. Показываются последние 100 обращений.")
    st.button("Обновить список")
    try:
        tickets = support_store.list_tickets()
        if not tickets:
            st.info("Пока нет обращений. Их можно создать после ответа ассистента.")
        for ticket in tickets:
            status = "Новое" if ticket["status"] == "new" else "Закрыто"
            with st.expander(f"№{ticket['id']} · {status} · {ticket['question'][:90]}"):
                st.write(ticket["question"])
                st.caption(f"Создано: {ticket['created_at']} (UTC)")
                response = json.loads(ticket["result_json"])
                st.write(response["answer"])
                st.caption(f"Результат обработки: {response['status']}")
                for hit in json.loads(ticket["evidence_json"]):
                    st.link_button(hit["title"], hit["source_url"], key=f"link_{ticket['id']}_{hit['chunk_id']}")
                    st.text(hit["text"])
                if ticket["status"] == "new" and st.button("Закрыть обращение", key=f"close_{ticket['id']}"):
                    support_store.close_ticket(ticket["id"])
                    st.rerun()
    except sqlite3.Error:
        st.error("Не удалось открыть очередь обращений.")
