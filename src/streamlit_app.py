import streamlit as st
import sys

sys.path.append("src")

from query import ask_question

st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="🤖"
)

st.title("🤖 Document Q&A Bot")

st.write(
    "Ask questions about the uploaded documents."
)

question = st.text_input(
    "Enter your question"
)

if st.button("Ask"):

    if question:

        answer, sources = ask_question(question)

        st.subheader("Answer")
        st.write(answer)

        if sources:

            st.subheader("Sources")

            for source in sources:
                st.write(f"• {source}")