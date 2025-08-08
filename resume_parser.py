import fitz  # PyMuPDF
import spacy
import spacy
import subprocess
import streamlit as st

@st.cache_resource
def get_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")


def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_candidate_name(resume_text):
    nlp = get_spacy_model()

    lines = [line.strip() for line in resume_text.strip().split('\n') if line.strip()]
    section_keywords = {
        'contact', 'summary', 'profile', 'linkedin', 'email', 'phone', 'location',
        'experience', 'professional', 'skills', 'education', 'work', 'objective',
        'portfolio', 'github'
    }
    job_titles = {
        'engineer', 'developer', 'manager', 'intern', 'consultant', 'scientist',
        'analyst', 'designer', 'officer', 'leader', 'executive', 'architect'
    }
    known_locations = {
        "dallas", "houston", "austin", "texas", "new york", "los angeles", "california",
        "boston", "seattle", "san francisco", "atlanta", "chicago", "miami", "denver",
        "washington", "phoenix", "san jose", "college station", "tx", "ny", "ca"
    }

    def likely_title(line):
        line_low = line.lower()
        return any(title in line_low for title in job_titles) and len(line.split()) <= 4

    def is_name_like(line):
        words = line.split()
        line_lower = line.lower()
        return (
            1 < len(words) <= 4 and
            all(word[0].isupper() or word.isupper() for word in words) and
            not any(kw in line_lower for kw in section_keywords) and
            not likely_title(line) and
            not any(loc in line_lower for loc in known_locations)
        )

    candidates = []

    for line in lines[:12]:
        if is_name_like(line):
            candidates.append(line)
    for line in reversed(lines[-20:]):
        if is_name_like(line) and line not in candidates:
            candidates.append(line)

    if candidates:
        return candidates[0]

    try:
        doc = nlp(resume_text)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if names:
            return names[0]
    except:
        pass

    for line in lines:
        if not any(kw in line.lower() for kw in section_keywords):
            return line

    return "Unknown"
