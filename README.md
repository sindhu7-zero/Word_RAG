# 📄 Document Search and Summarization using RAG (LLM)

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that enables users to **search across PDF documents and generate concise summaries** using a Large Language Model (LLM).

The system combines **traditional information retrieval (TF-IDF)** with **semantic search (FAISS + embeddings)** to improve retrieval accuracy and robustness. A **Streamlit web interface** is provided for easy interaction.

This project was developed as part of an **interview assignment** focused on document search, summarization, scalability, and correctness.

---

## Features

- 📚 PDF document ingestion  
- 🧹 Text cleaning and chunking  
- 🔍 Hybrid retrieval (TF-IDF + semantic embeddings)  
- 🧠 LLM-based summarization (Groq – LLaMA 3)  
- 📏 Adjustable summary length  
- 🚑 Safe fallback for noisy or scanned PDFs  
- 🖥️ Simple and clean Streamlit UI  
- 🔐 Environment variable–based API key management  

---

## Tech Stack

- **Python**
- **LangChain**
- **Groq LLM (LLaMA-3.3-70B)**
- **FAISS**
- **Sentence Transformers**
- **Scikit-learn (TF-IDF)**
- **Streamlit**

---



