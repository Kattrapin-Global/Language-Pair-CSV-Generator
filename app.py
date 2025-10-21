import io
from typing import List, Tuple

import streamlit as st
import pandas as pd

# File readers
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

# NLP
import spacy
from spacy.language import Language
from spacy.cli import download as spacy_download

# Sentence embeddings
from sentence_transformers import SentenceTransformer, util


# -------------------------------
# Helpers
# -------------------------------
SUPPORTED_EXTS = {"txt", "pdf", "docx"}

LANG_NAME_TO_CODE = {
    # Common mappings (extend as needed)
    "japanese": "ja",
    "chinese": "zh",
    "mandarin": "zh",
    "simplified chinese": "zh",
    "traditional chinese": "zh",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "tamil": "ta",
    "telugu": "te",
    "marathi": "mr",
    "bengali": "bn",
    "italian": "it",
    "portuguese": "pt",
    "turkish": "tr",
    "thai": "th",
    "vietnamese": "vi",
}

SPACY_MODEL_BY_CODE = {
    # Known small pipelines. Try these first; otherwise fallback to blank + sentencizer.
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm",
    "ru": "ru_core_news_sm",
    "ro": "ro_core_news_sm",
}

@st.cache_resource(show_spinner=False)
def get_embedder():
    # Multilingual sentence embeddings model
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def normalize_language_input(text: str) -> Tuple[str, str]:
    """Return (lang_label, lang_code_guess)."""
    if not text:
        return ("Foreign", "xx")
    label = text.strip()
    low = label.lower()
    # If user gave an ISO-ish code
    if len(low) in (2, 3) and low.isalpha():
        return (label, low)
    # Map common names to codes
    code = LANG_NAME_TO_CODE.get(low, "xx")
    return (label, code)


def load_spacy_pipeline(lang_code: str, is_english: bool = False) -> Language:
    """Load a spaCy pipeline with safe fallbacks.
    Try language-specific small model; else fallback to blank + sentencizer.
    """
    if is_english:
        target_model = SPACY_MODEL_BY_CODE.get("en", "en_core_web_sm")
    else:
        target_model = SPACY_MODEL_BY_CODE.get(lang_code)

    nlp = None
    if target_model:
        try:
            nlp = spacy.load(target_model)
        except OSError:
            # Attempt to download, then load
            try:
                spacy_download(target_model)
                nlp = spacy.load(target_model)
            except Exception:
                nlp = None

    if nlp is None:
        # Fallback: blank pipeline with rule-based sentence boundary detection
        try:
            nlp = spacy.blank("en" if is_english else (lang_code if lang_code != "xx" else "xx"))
        except Exception:
            # Last-resort universal blank
            nlp = spacy.blank("xx")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    return nlp


def read_uploaded_text(file) -> str:
    """Extract plain text from an uploaded file-like object (txt/pdf/docx)."""
    if file is None:
        return ""
    name = file.name
    ext = name.split(".")[-1].lower()

    try:
        if ext == "txt":
            # Try utf-8; fallback to latin-1
            try:
                return file.read().decode("utf-8")
            except Exception:
                file.seek(0)
                return file.read().decode("latin-1", errors="ignore")
        elif ext == "pdf":
            file.seek(0)
            data = file.read()
            return pdf_extract_text(io.BytesIO(data)) or ""
        elif ext == "docx":
            file.seek(0)
            doc = Document(io.BytesIO(file.read()))
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return ""
    except Exception as e:
        raise RuntimeError(f"Failed to read {name}: {e}")


def segment_sentences(nlp: Language, text: str) -> List[str]:
    if not text:
        return []
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents]
    # Clean and drop empties
    sents = [s for s in (s.strip() for s in sents) if s]
    return sents


def greedy_align_by_similarity(src_sents: List[str], tgt_sents: List[str]) -> List[Tuple[str, str]]:
    """Greedy 1-1 alignment using cosine similarity on SentenceTransformer embeddings.
    Maps each source sentence to its best unused target sentence.
    """
    if not src_sents or not tgt_sents:
        return []

    embedder = get_embedder()
    src_emb = embedder.encode(src_sents, convert_to_tensor=True, show_progress_bar=False)
    tgt_emb = embedder.encode(tgt_sents, convert_to_tensor=True, show_progress_bar=False)

    sim = util.cos_sim(src_emb, tgt_emb).cpu().tolist()  # shape: [len(src), len(tgt)]

    used_tgt = set()
    aligned = []
    for i, row in enumerate(sim):
        # pick the best target not yet used
        best_j, best_val = -1, -1.0
        for j, val in enumerate(row):
            if j in used_tgt:
                continue
            if val > best_val:
                best_j, best_val = j, val
        if best_j >= 0:
            used_tgt.add(best_j)
            aligned.append((src_sents[i], tgt_sents[best_j]))
        # else: skip if none available
    return aligned


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Language Pair CSV Generator", layout="wide")
st.title("Language Pair CSV Generator")
st.write(
    "Upload a document in any language and its English translation to generate a bilingual CSV dataset."
)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Step 1: Foreign-language file")
    foreign_file = st.file_uploader(
        "Upload foreign-language file (TXT, PDF, or DOCX)",
        type=list(SUPPORTED_EXTS),
        key="foreign_upload",
    )
    lang_input = st.text_input(
        "Enter the language name or code (e.g., Japanese, ja)",
        key="lang_input",
        placeholder="e.g., Japanese or ja",
    )

with col_right:
    st.subheader("Step 2: English file")
    english_file = st.file_uploader(
        "Upload corresponding English file (TXT, PDF, or DOCX)",
        type=list(SUPPORTED_EXTS),
        key="english_upload",
    )

can_generate = bool(foreign_file and english_file and (lang_input or "").strip())

st.divider()

if st.button("Generate CSV", type="primary", disabled=not can_generate):
    if not foreign_file or not english_file:
        st.warning("Please upload both files before generating the CSV.")
        st.stop()

    lang_label, lang_code = normalize_language_input(lang_input)

    try:
        with st.status("Reading files…", expanded=False) as status:
            foreign_text = read_uploaded_text(foreign_file)
            english_text = read_uploaded_text(english_file)
            if not foreign_text:
                st.error("Could not extract text from the foreign-language file.")
                st.stop()
            if not english_text:
                st.error("Could not extract text from the English file.")
                st.stop()
            status.update(label="Files read successfully.")
    except Exception as e:
        st.error(f"Error reading files: {e}")
        st.stop()

    with st.status("Loading NLP pipelines…", expanded=False) as status:
        nlp_en = load_spacy_pipeline("en", is_english=True)
        nlp_src = load_spacy_pipeline(lang_code or "xx", is_english=False)
        status.update(label="NLP ready.")

    with st.status("Segmenting sentences…", expanded=False) as status:
        src_sents = segment_sentences(nlp_src, foreign_text)
        en_sents = segment_sentences(nlp_en, english_text)
        status.update(label=f"Found {len(src_sents)} {lang_label} sentences and {len(en_sents)} English sentences.")

    if not src_sents or not en_sents:
        st.error("No sentences detected. Ensure the files contain readable text.")
        st.stop()

    # Basic alignment
    with st.status("Aligning sentences… (semantic)", expanded=False) as status:
        if len(src_sents) == len(en_sents):
            aligned_pairs = list(zip(src_sents, en_sents))
        else:
            aligned_pairs = greedy_align_by_similarity(src_sents, en_sents)
        status.update(label=f"Aligned {len(aligned_pairs)} sentence pairs.")

    if len(en_sents) < len(src_sents):
        st.warning(
            f"English file seems shorter: {len(en_sents)} vs {len(src_sents)}. "
            f"{len(src_sents) - len(aligned_pairs)} sentences could not be aligned."
        )

    # Build DataFrame
    if aligned_pairs:
        df = pd.DataFrame(aligned_pairs, columns=[lang_label, "English"])
    else:
        df = pd.DataFrame(columns=[lang_label, "English"])

    st.subheader("Preview (first 10 aligned pairs)")
    st.dataframe(df.head(10), use_container_width=True, height=320)

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    filename = f"{lang_label.replace(' ', '_')}_to_en.csv"
    st.download_button("Download CSV", data=csv_bytes, file_name=filename, mime="text/csv")

st.divider()
st.caption("Generated bilingual datasets can be used to fine-tune translation or alignment models later.")
