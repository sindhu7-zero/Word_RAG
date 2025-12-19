import streamlit as st
import os
from rag import run_rag

st.set_page_config(page_title="RAG Document Search", layout="centered")

st.title("📄 Document Search & Summarization")
st.write("Upload PDFs and ask questions to get concise summaries.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

query = st.text_input("Enter your query")

summary_length = st.slider(
    "Summary length (words)",
    min_value=50,
    max_value=300,
    value=120
)

if st.button("Search & Summarize"):
    if not uploaded_files or not query:
        st.warning("Please upload PDFs and enter a query.")
    else:
        os.makedirs("uploads", exist_ok=True)
        pdf_paths = []

        for file in uploaded_files:
            path = os.path.join("uploads", file.name)
            with open(path, "wb") as f:
                f.write(file.read())
            pdf_paths.append(path)

        with st.spinner("Processing documents..."):
            summary, sources = run_rag(pdf_paths, query, summary_length)

        st.subheader("📌 Summary")
        st.write(summary)

        # if sources:
        #     st.subheader("📚 Source Excerpts")
        #     for i, doc in enumerate(sources, 1):
        #         st.markdown(f"**{i}.** {doc.page_content[:250]}...")
