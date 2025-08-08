import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

from pinecone_utils import get_or_create_index, upsert_resume_embedding, query_similar_resumes, clear_all_resumes
from embedding_utils import get_embedding_model


def index_resumes_fresh(resumes, model):
    index = get_or_create_index(dimension=model.get_sentence_embedding_dimension())
    clear_all_resumes(index)

    for i, resume in enumerate(resumes):
        emb = model.encode([f"passage: {resume['text']}"], normalize_embeddings=True)[0]
        metadata = {
            "name": resume["name"],
            "filename": resume["filename"],
            "text": resume["text"]
        }
        upsert_resume_embedding(index, emb.tolist(), metadata, resume['filename'])

    return index


def rank_candidates_from_pinecone(job_desc, model, top_k):
    index = get_or_create_index(dimension=model.get_sentence_embedding_dimension())

    job_emb = model.encode([f"query: {job_desc}"], normalize_embeddings=True)[0]
    results = query_similar_resumes(index, job_emb.tolist(), top_k=top_k)

    data = []
    for match in results.matches:
        metadata = match.metadata
        data.append({
            "Candidate Name": metadata.get("name", "Unknown"),
            "File Name": metadata.get("filename", "Unknown"),
            "Similarity Score": round(match.score, 4),
            "text": metadata.get("text", "")
        })

    return pd.DataFrame(data)


def generate_batch_fit_summaries(candidates_data, job_description):
    hf_token = st.secrets["HF_TOKEN"]
    login(token=hf_token)

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto", use_auth_token=hf_token,
        trust_remote_code=True
    )

    summarizer = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=400, temperature=0.2, top_p=0.85,
        repetition_penalty=1.15, do_sample=True,
    )

    summaries = []
    for _, row in candidates_data.iterrows():
        resume = row["text"]
        name = row["Candidate Name"]
        score = row["Similarity Score"]

        prompt = (
            "You are an AI assistant helping a recruiter. Your task is to generate a concise, factual, and detailed summary "
            "explaining why a candidate is a good fit for a job. The summary should be 3–4 sentences long. "
            "Use only the content that appears in the resume. Do not copy or repeat the job description. "
            "If relevant content is not found in the resume, say: 'The resume does not contain enough information to determine fit.'\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Candidate Resume:\n{resume}\n\n"
            "Summary:"
        )

        try:
            output = summarizer(prompt)[0]['generated_text']
            summary = output[len(prompt):].strip() if output.startswith(prompt) else output.strip()
            summary = re.sub(r"(?i)(summary:|answer:|response:)", "", summary).strip()
            summary = summary.split('\n\n')[0]
            summary = summary.split('\n')[0] if summary.count('\n') > 4 else summary

            if len(summary) < 40 or "I am an AI" in summary:
                summary = "The resume does not contain enough information to determine fit."

            summaries.append(summary)

        except Exception as e:
            print(f"Error generating summary for {name}: {e}")
            summaries.append("Unable to generate summary.")

    return summaries
