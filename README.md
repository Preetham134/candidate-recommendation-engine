![App Preview](https://github.com/Preetham134/candidate-recommendation-engine/blob/main/Engine.png)
![App Preview](https://github.com/Preetham134/candidate-recommendation-engine/blob/main/Summaries.png)

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

1. 📝 Upload a **Job Description (.txt)**.
2. 📎 Upload multiple **Resumes (.pdf)**.
3. 📊 App embeds and indexes resumes via **Pinecone**.
4. 🧠 Job description is embedded and **Top-K candidates** are retrieved.
5. 🤖 **LLM** generates a concise fit summary for each.
6. ✅ Results are presented in the Streamlit UI.

> ⚠️ **Note**: Due to external latency (API, model loading, etc.), if you **change the Top-N value or upload new resumes or job descriptions**, results may not update immediately.  
> If you don’t see results, wait a few seconds and **try clicking "Process" again**. Avoid rapidly pressing the button.

---

## ☁️ Why Pinecone, Transformers, and LLMs?

### ✅ **Pinecone** – *Scalable Vector Search*
- Handles **millions of high-dimensional vectors** with blazing speed.
- Offers **persistent storage**, so you don’t need to re-index every time.
- Allows **real-time retrieval** with configurable top-k matching.
- Offloads the complexity of managing your own similarity search infra (e.g., FAISS).
- Enables rapid prototyping **without infrastructure headaches**.

### ✅ **Transformers (Sentence Transformers)** – *Contextual Embedding*
- `intfloat/e5-large-v2` is **fine-tuned for search tasks** using contrastive learning.
- Captures **meaningful sentence-level semantics** over just keyword overlap.

### ✅ **LLM (LLaMA 3)** – *Natural Language Understanding*
- Can **summarize** large amounts of resume text into **short, factual fit reports**.
- Adds **explainability** behind semantic matches.

---

## 🌍 Deployment Guide

### ✅ Local Setup

Only three steps:

```bash
# 1. Clone the repo
git clone https://github.com/your-username/candidate-recommender.git
cd candidate-recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> ⚠️ **Note on Hosting**:
- Attempted hosting on **Streamlit Cloud**, but faced issues with **dependency resolution failures** (e.g., en_core_web_sm).
  Link: https://candidate-recommendation-engine-ngs2xkgms66oatrwtalkbo.streamlit.app/
- Tried other online providers, but **RAM limitations** (e.g., <4GB) were insufficient for LLaMA inference and embedding generation.
- **Local usage** with adequate memory and internet is currently **recommended**.

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

Before running, ensure:

```bash
python -m spacy download en_core_web_sm
```

---

## 📌 Future Enhancements

- 🔄 Add resume **chunking** to handle long documents (avoid embedding cutoffs).
- 📊 Integrate **charts and analytics** to visually compare candidate scores.
- 📃 Enable **CSV or PDF export** of results.
- 💬 Add a **chatbot assistant** for recruiter queries (e.g., “Who is the strongest match?”).
- 🧠 Use **LangChain/LangGraph** to enrich or control LLM behavior more dynamically.
- 🚀 **Dockerize and deploy** on AWS/GCP/Azure for production readiness.
- 🧵 **Reduce the number of LLM calls** (e.g., batch or cache summaries) to improve speed and cost.
