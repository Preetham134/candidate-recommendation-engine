import streamlit as st 
from resume_parser import extract_text_from_pdf, extract_candidate_name
from matching import generate_batch_fit_summaries, index_resumes_fresh, rank_candidates_from_pinecone
from embedding_utils import get_embedding_model
import pandas as pd

st.set_page_config(page_title="Candidate Recommendation Engine", layout="wide")

st.markdown("<h1 style='text-align: center;'>🔍 Candidate Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 📄 Upload Job Description")
    job_file = st.file_uploader("Upload a Job Description (.txt)", type=['txt'])

    st.markdown("### 📂 Upload Resumes")
    resume_files = st.file_uploader("Upload one or more resumes (.pdf)", type=["pdf"], accept_multiple_files=True)

    top_k = st.selectbox("🔢 Select number of top candidates to view:", options=list(range(1, 11)), index=2)

    process_button = st.button("🚀 Process & Recommend Candidates")

if process_button and job_file and resume_files:
    with st.spinner("🔄 Processing job description and resumes... Please wait."):
        job_desc_text = job_file.read().decode('utf-8')

        resumes = []
        for uploaded_file in resume_files:
            text = extract_text_from_pdf(uploaded_file)
            name = extract_candidate_name(text)
            resumes.append({
                "name": name,
                "filename": uploaded_file.name,
                "text": text
            })

        st.write(f"**📄 Processing {len(resumes)} resumes:**")
        model = get_embedding_model()
        
        index_resumes_fresh(resumes, model)
        df = rank_candidates_from_pinecone(job_desc_text, model, top_k=top_k)
        top_df = df.copy()

        st.write("**🎯 Pinecone Query Results (Top Candidates):**")
        for _, row in top_df.iterrows():
            st.write(f"- **{row['Candidate Name']}** ({row['File Name']}) - Score: {row['Similarity Score']}")

        st.markdown("## 🤖 AI Fit Summaries")
        summaries = generate_batch_fit_summaries(top_df, job_desc_text)
        top_df['AI Fit Summary'] = summaries

        for _, row in top_df.iterrows():
            st.markdown(f"""
            <div style='border: 2px solid #e6e6e6; padding: 15px; border-radius: 10px; margin-bottom: 20px; background-color: #f9f9f9;'>
                <h4>{row['Candidate Name']} 📎</h4>
                <p><strong>Similarity Score:</strong> {row['Similarity Score']}</p>
                <p><strong>Fit Summary:</strong><br> {row['AI Fit Summary']}</p>
            </div>
            """, unsafe_allow_html=True)

elif process_button and (not job_file or not resume_files):
    st.warning("⚠️ Please upload both a job description and at least one resume before processing.")
else:
    st.info("⬆️ Please upload a job description and resumes, then click **Process & Recommend Candidates**.")
