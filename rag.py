# ==========================================
# RAG Pipeline: Document Search & Summarization
# ==========================================

import os
import re
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# 1️⃣ Text Cleaning
# ----------------------------------
def clean_text(text):
    text = text.lower()  # lowercase for consistency
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    text = re.sub(r"[^a-z0-9\s\-]", "", text)  # keep hyphens
    return text.strip()

# ----------------------------------
# 2️⃣ Load & Split PDFs
# ----------------------------------
def load_documents(pdf_paths):
    documents = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            cleaned = clean_text(page.page_content)
            if cleaned:
                page.page_content = cleaned
                documents.append(page)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # Safety filter: ignore very short chunks
    return [c for c in chunks if len(c.page_content.split()) > 5]

# ----------------------------------
# 3️⃣ Hybrid Retriever (TF-IDF + Embeddings)
# ----------------------------------
class HybridRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.texts = [d.page_content for d in docs]
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.tfidf = TfidfVectorizer(stop_words="english")
        try:
            self.tfidf_matrix = self.tfidf.fit_transform(self.texts)
        except ValueError:
            self.tfidf_matrix = None

    def search(self, query, k=5):
        query_clean = clean_text(query)
        # Try TF-IDF first
        if self.tfidf_matrix is not None:
            query_vec = self.tfidf.transform([query_clean])
            scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
            top_idx = scores.argsort()[-20:][::-1]  # take top 20 candidates
            candidate_docs = [self.docs[i] for i in top_idx]
            temp_store = FAISS.from_documents(candidate_docs, self.embeddings)
            return temp_store.similarity_search(query_clean, k=k)

        # fallback to embeddings if TF-IDF fails
        return self.vectorstore.similarity_search(query_clean, k=k)

# ----------------------------------
# 4️⃣ Summarization with LLM
# ----------------------------------
def summarize(docs, query, length=150):
    context = "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_template("""
You are a document-based assistant.

Rules:
1. Use the context to answer.
2. If the context does not clearly cover the question:
   - Say: "The documents may not contain complete information."
   - Then provide a brief general explanation.

Context:
{context}

Question:
{query}

Answer:
""")

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "query": query,
        "length": length
    })

# ----------------------------------
# 5️⃣ Full RAG Pipeline
# ----------------------------------
def run_rag(pdf_paths, query, summary_length=150):
    docs = load_documents(pdf_paths)
    if not docs:
        return "No readable text found in uploaded PDFs.", []

    retriever = HybridRetriever(docs)
    results = retriever.search(query, k=5)

    # Debug: show retrieved docs preview
    print("\nRetrieved Documents Preview:")
    for i, doc in enumerate(results):
        print(f"Doc {i}:", doc.page_content[:200], "...")

    summary = summarize(results, query, summary_length)
    return summary, results

# ----------------------------------
# 6️⃣ Example Usage
# ----------------------------------
if __name__ == "__main__":
    pdf_paths = ["sample1.pdf", "sample2.pdf"]  # replace with your PDFs
    query = "cross-validation in machine learning"
    summary_length = 150

    summary, retrieved_docs = run_rag(pdf_paths, query, summary_length)

    print("\n===== Summary =====")
    print(summary)
