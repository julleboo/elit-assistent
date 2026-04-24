import os
import re
import urllib.parse
import logging
import hashlib
import time
import streamlit as st
from google.cloud import discoveryengine
from google.api_core.client_options import ClientOptions
from google.oauth2 import service_account

# --- LOGGNING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- KONFIGURATION (Hämtas från st.secrets i Streamlit Cloud eller .env lokalt) ---
def load_config() -> dict:
    """Läser in konfiguration och validerar obligatoriska nycklar."""
    # Prioritera Streamlit Secrets, annars os.environ
    config = {
        "project_id": st.secrets.get("GOOGLE_PROJECT_ID") or os.getenv("GOOGLE_PROJECT_ID"),
        "engine_id": st.secrets.get("GOOGLE_ENGINE_ID") or os.getenv("GOOGLE_ENGINE_ID"),
        "bucket_url": st.secrets.get("BUCKET_URL") or os.getenv("BUCKET_URL"),
    }
    
    missing = [k for k, v in config.items() if not v]
    if missing:
        st.error(f"Saknade miljövariabler: {', '.join(missing)}")
        st.stop()
    return config

CONFIG = load_config()

# --- KONSTANTER ---
SEARCH_RESULT_COUNT = 5
SUMMARY_RESULT_COUNT = 5
SUFFICIENT_SCORE_THRESHOLD = 8.0 
RATE_LIMIT_SECONDS = 2

_RE_CITATION = re.compile(r'\[\d+(?:,\s*\d+)*\]')
_RE_PDF_SUFFIX = re.compile(r'\.pdf.*$', flags=re.IGNORECASE)

STOP_WORDS = {"technical", "data", "type", "page", "table", "the", "and", "for"}
NEGATIVE_WORDS = {"customer", "service", "contact", "email", "representative", "http", "www", "call", "inquiries"}

# --- SINGLETON-KLIENT ---
@st.cache_resource
def get_search_client():
    try:
        client_options = ClientOptions(api_endpoint="discoveryengine.googleapis.com")
        if "GOOGLE_CREDENTIALS" in st.secrets:
            creds = service_account.Credentials.from_service_account_info(
                dict(st.secrets["GOOGLE_CREDENTIALS"]),
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            return discoveryengine.SearchServiceClient(credentials=creds, client_options=client_options)
        return discoveryengine.SearchServiceClient(client_options=client_options)
    except Exception as e:
        st.error(f"Kunde inte initiera Vertex AI: {e}")
        st.stop()

# --- HJÄLPFUNKTIONER (Från logic.py) ---

def fix_smashed_words(text: str) -> str:
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\))([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\()', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def format_title(uri: str) -> str:
    file_name = uri.split("/")[-1]
    return _RE_PDF_SUFFIX.sub('', file_name).replace('-', ' ').replace('_', ' ').title()

def build_pdf_url(uri: str, page: int) -> str:
    safe_name = urllib.parse.quote(uri.split("/")[-1])
    return f"{CONFIG['bucket_url']}/{safe_name}#page={page}"

def get_verified_extract(result, evidence_quotes: list[str], display_answer: str, query: str) -> tuple[str, float]:
    chunk = result.chunk
    content = chunk.content
    
    # Grundstädning
    clean_text = content.replace("...", " till ").replace("#", "")
    noise = ["_START_OF_TABLE_", "_END_OF_TABLE_", "TABLE_IN_MARKDOWN:", "|", "__"]
    for n in noise:
        clean_text = clean_text.replace(n, " ")
    
    # Fixa smashed words i råtexten först
    clean_text = fix_smashed_words(clean_text)
    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n', clean_text) if len(s.strip()) > 10]

    # Gatekeeper (Produktkontroll)
    target_products = set(re.findall(r'[A-Z]{1,3}\d{1,4}(?:-[A-Z\d]+)?|\d{4}-\w+', display_answer + " " + query))
    if target_products and not any(p.upper() in content.upper() for p in target_products):
        return "", 0.0

    product_numbers = set(re.findall(r'\d+', " ".join(target_products)))    

    # Scoring
    weights = {}
    for quote in evidence_quotes:
        cleaned_quote = fix_smashed_words(quote)
        nums = re.findall(r'\d+[.,]\d+|\d+', cleaned_quote)
        for n in nums:
            if n not in product_numbers: weights[n.replace(',', '.')] = 5.0
        words = re.findall(r'[a-zA-Z]{4,}', cleaned_quote.lower())
        for w in words:
            if w not in STOP_WORDS: weights[w] = 1.0

    requires_numbers = any(v == 5.0 for v in weights.values())
    best_sentence = ""
    max_score = 0.0
    
    for sentence in sentences:
        if any(neg in sentence.lower() for neg in NEGATIVE_WORDS): continue 
        w_score = sum(1.0 for t, w in weights.items() if w == 1.0 and t in sentence.lower())
        n_score = sum(5.0 for t, w in weights.items() if w == 5.0 and t in sentence.replace(',', '.'))
        
        if requires_numbers and n_score == 0: continue
        total = w_score + n_score
        if total > max_score:
            max_score = total
            best_sentence = sentence

    if max_score < 4.0 or not best_sentence: return "", 0.0

    # FORMATERING: Markerpenna med HTML
    display_text = best_sentence
    sorted_tokens = sorted(weights.keys(), key=len, reverse=True)
    for token in sorted_tokens:
        if len(token) >= 2:
            pattern = re.compile(rf'\b({re.escape(token)})\b', re.IGNORECASE)
            # Vi lägger till ett mellanslag i ersättningen för att vara säkra
            display_text = pattern.sub(r'<mark style="background-color: #FFFF00; color: black; padding: 0 2px; border-radius: 2px;">\1</mark>', display_text)
    
    # FIX FÖR WORD SMASHING: Slå ihop markeringar men BEHÅLL mellanslag/tecken emellan
    # Vi letar efter slutet på en markering och början på nästa, och ser till att \1 (mellanrummet) bevaras
    display_text = re.sub(r'</mark>(\s*[.,:;-]?\s*)<mark[^>]*>', r'\1', display_text)
    
    return display_text.strip(), max_score

# --- CORE SEARCH ENGINE ---

def run_elit_search(query: str):
    client = get_search_client()
    serving_config = f"projects/{CONFIG['project_id']}/locations/global/collections/default_collection/engines/{CONFIG['engine_id']}/servingConfigs/default_search"

    summary_spec = discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
        summary_result_count=SUMMARY_RESULT_COUNT,
        include_citations=True,
        model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
            preamble="""Du är en teknisk expertassistent för Elit Instrument.
1. SVARA ALLTID PÅ SVENSKA.
2. Var saklig och kortfattad, svara bara på det användaren frågar efter.
3. Om du är osäker, GISSA ALDRIG. säg isåfall 'Kan ej fastställas i nuläget, kontakta Elit:s support'.
4. BEVISKRAV: Efter ditt svar, lägg en rad med '---' och lista de viktigaste tekniska bevisen på det språket som står i manualen. (exempel: lista på engelska om det är engelsk manual)
"""
        )
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=SEARCH_RESULT_COUNT,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            search_result_mode=discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS,
            summary_spec=summary_spec,
            extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(max_extractive_segment_count=1)
        )
    )

    response = client.search(request)
    if not response.summary or not response.summary.summary_text:
        return {"answer": "Informationen kunde inte fastställas.", "sources": []}

    raw_text = fix_smashed_words(_RE_CITATION.sub('', response.summary.summary_text))
    if '---' in raw_text:
        parts = raw_text.split('---')
        display_answer, evidence_quotes = parts[0].strip(), [l.strip('- *') for l in parts[1].strip().split('\n') if len(l.strip()) > 1]
    else:
        display_answer, evidence_quotes = raw_text.strip(), re.findall(r'\d+[.,]?\d*|[A-Z]{2,}\d+', raw_text)

    all_candidates = []
    shown_hashes = set()

    for result in response.results:
        extract, score = get_verified_extract(result, evidence_quotes, display_answer, query)
        if extract and score >= 4.0:
            e_hash = hashlib.md5(extract.encode()).hexdigest()
            if e_hash not in shown_hashes:
                shown_hashes.add(e_hash)
                source_obj = {
                    "score": score,
                    "data": {
                        "title": format_title(result.chunk.document_metadata.uri),
                        "extract": extract,
                        "page": getattr(result.chunk.page_span, 'page_start', 1),
                        "url": build_pdf_url(result.chunk.document_metadata.uri, getattr(result.chunk.page_span, 'page_start', 1))
                    }
                }
                if score >= SUFFICIENT_SCORE_THRESHOLD:
                    return {"answer": display_answer, "sources": [source_obj["data"]]}
                all_candidates.append(source_obj)

    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    return {
        "answer": display_answer,
        "sources": [c["data"] for c in all_candidates[:2]]
    }

# --- STREAMLIT UI ---

st.set_page_config(page_title="Elit AI Expert", page_icon="⚡")

# Header
st.markdown(
    """
    <div style="background-color: white; padding: 10px; border-radius: 5px; display: inline-block; margin-bottom: 20px;">
        <img src="https://elit.se/wp-content/uploads/2022/12/elit-logo-svart-transp-420x113-1.svg" width="200">
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Teknisk Support AI")
st.markdown("Expertstöd baserat på officiella manualer och datablad.")

query = st.text_input("Fråga här:", placeholder="T.ex. Hur ofta behöver TP M255S kalibreras?")

if query:
    # Rate limiting
    now = time.monotonic()
    if now - st.session_state.get("last_search", 0) < RATE_LIMIT_SECONDS:
        st.warning("Vänligen vänta ett ögonblick innan nästa sökning.")
        st.stop()
    st.session_state["last_search"] = now

    with st.spinner("Analyserar dokumentation..."):
        result = run_elit_search(query)

    st.subheader("Svar")
    st.write(result["answer"])
    st.divider()

    st.subheader("Verifierade källor")
    if not result["sources"]:
        st.warning("⚠️ Svaret kunde inte styrkas ordagrant i källmaterialet.")
    else:
        for src in result["sources"]:
            with st.container(border=True):
                # Titel och sidnummer
                st.markdown(f"📄 **{src['title']}** (Sid. {src['page']})")
                
                # Utdraget med markerpenna (HTML tillåten)
                with st.expander("Visa tekniskt utdrag", expanded=True):
                    st.markdown(f"<div style='line-height: 1.6;'>{src['extract']}</div>", unsafe_allow_html=True)
                
                # KNAPPEN SOM FÖRSVANN:
                st.link_button(f"Öppna PDF på sida {src['page']}", src['url'])