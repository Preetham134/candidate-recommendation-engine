# 🔍 Candidate Recommendation Engine

A powerful AI-powered web application that takes in a **Job Description** and multiple **PDF Resumes**, and returns the **Top Matching Candidates** ranked by semantic relevance, along with **LLM-generated fit summaries**. Built using `Streamlit`, `Sentence Transformers`, `LLaMA 3`, and `Pinecone`.

---

## 📚 Table of Contents

- [🚀 Overview](#-overview)
- [🧠 Tech Stack](#-tech-stack)
- [🏗️ System Architecture](#-system-architecture)
- [⚙️ Implementation Details](#️-implementation-details)
- [🔁 Workflow](#-workflow)
- [☁️ Why Pinecone, Transformers, and LLMs?](#️-why-pinecone-transformers-and-llms)
- [🌍 Deployment Guide](#-deployment-guide)
- [📦 Requirements](#-requirements)
- [📌 Future Enhancements](#-future-enhancements)
- [🧑‍💻 Author](#-author)

---

## 🚀 Overview

This application helps recruiters, hiring managers, and HR professionals **automatically match candidates to job roles** by:

1. Extracting resume text from PDF files.
2. Computing semantic similarity scores between resumes and job descriptions using embeddings.
3. Ranking resumes using a **Pinecone-powered vector database**.
4. Generating natural language **AI fit summaries** using **LLaMA 3**.

It's built for scalability, fast processing, and high accuracy in candidate-job matching.

---

## 🧠 Tech Stack

| Layer            | Tools / Libraries                                      |
|------------------|--------------------------------------------------------|
| **Frontend**     | Streamlit (Python-based Web UI)                        |
| **Embedding**    | Sentence Transformers (`intfloat/e5-large-v2`)         |
| **Vector Search**| Pinecone                                                |
| **LLM Summary**  | LLaMA 3 (Meta) via HuggingFace                         |
| **PDF Parsing**  | PyMuPDF (`fitz`)                                       |
| **NER (Name)**   | spaCy (`en_core_web_sm`)                               |
| **Deployment**   | Ngrok (for local tunneling) / Streamlit Share / AWS    |

---

## 🏗️ System Architecture

```plaintext
[User Uploads]
  ├── Job Description (.txt)
  └── Resume PDFs (.pdf)
         ↓
 [Text Extraction]
  ├── extract_text_from_pdf()
  └── extract_candidate_name()
         ↓
 [Embedding Model: Sentence Transformers]
         ↓
 [Pinecone Vector DB]
  ├── Upsert resume embeddings
  └── Query top-K matches with job embedding
         ↓
 [Ranking Output]
         ↓
 [LLM (LLaMA-3) Fit Summary Generation]
         ↓
 [Streamlit Frontend Display]
```

---

## ⚙️ Implementation Details

### 1. **Text Extraction**  
`resume_parser.py` uses PyMuPDF to extract clean resume text and `spaCy` to estimate the candidate's name.

### 2. **Embeddings with Sentence Transformers**  
`embedding_utils.py` loads the `intfloat/e5-large-v2` model which supports the **"query: ..." and "passage: ..."** input format for better performance in information retrieval tasks.

### 3. **Resume Indexing**  
`matching.py::index_resumes_fresh()` clears previous embeddings and stores fresh ones in Pinecone. Metadata includes candidate name, filename, and raw text.

### 4. **Semantic Search with Pinecone**  
The job description is embedded and passed to Pinecone to retrieve **Top-K semantically similar resumes** using cosine similarity.

### 5. **Fit Summary Generation**  
`generate_batch_fit_summaries()` uses **Meta’s LLaMA 3 (3B)** hosted on Hugging Face to generate short, **factual, unbiased summaries** based purely on resume content.

---

## 🔁 Workflow

1. 📝 User uploads a **Job Description (.txt)**.
2. 📎 Uploads multiple **Resumes (.pdf)**.
3. 📊 The app embeds and indexes resumes via **Pinecone**.
4. 🧠 It embeds the job description and retrieves **Top-K candidates** using similarity.
5. 🤖 **LLM** generates a concise fit summary for each.
6. ✅ Results are presented on the Streamlit UI.

---

## ☁️ Why Pinecone, Transformers, and LLMs?

### ✅ **Pinecone** – *Scalable Vector Search*
- Handles **millions of high-dimensional embeddings** efficiently.
- Offers **real-time similarity search** with low latency.
- Eliminates the complexity of building your own FAISS index.

### ✅ **Transformers (Sentence Transformers)** – *Contextual Embedding*
- `intfloat/e5-large-v2` is **fine-tuned for semantic search**.
- Captures **contextual meaning** instead of keyword matching.
- Enhances **relevance ranking** accuracy for job-resume matching.

### ✅ **LLM (LLaMA 3)** – *Natural Language Understanding*
- Capable of **reasoning and summarizing** based on raw text.
- Generates **concise and tailored summaries** recruiters can read.
- Ensures **explainability** behind each recommendation.

---

## 🌍 Deployment Guide

### ✅ Local Setup
```bash
# 1. Clone the repo
git clone https://github.com/your-username/candidate-recommender.git
cd candidate-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit app
streamlit run app.py
```

### 🔐 API Keys
- **Pinecone**: Add your API key to `pinecone_utils.py`.
- **Hugging Face**: Add your token inside `matching.py`.

### 🌐 Ngrok (for public sharing)
```bash
pip install pyngrok
ngrok http 8501
```

> Then share the generated public URL.

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`:

```text
streamlit
sentence-transformers
PyMuPDF
scikit-learn
spacy
torch
transformers
huggingface_hub
requests
langchain_community
accelerate
pinecone
```

Make sure to run:
```bash
python -m spacy download en_core_web_sm
```

---

## 📌 Future Enhancements

- 🔄 Resume chunking for long documents (to avoid embedding truncation).
- 🧩 Use LangChain or LangGraph for advanced summary logic.
- 📊 Add charts for visual score comparison.
- 🗃️ Export results as CSV/PDF.
- 💬 Integrate chatbot to answer recruiter queries.
- 🚀 Dockerize and deploy on AWS/GCP/Azure.

---

## 🧑‍💻 Author

**Preetham Nandamuri**  
*Data Scientist | NLP | Generative AI | M.S. in Engineering Data Science*  
📫 [LinkedIn](https://www.linkedin.com/in/preethamnandamuri) | 🌐 Portfolio Coming Soon