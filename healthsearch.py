"""
HealthSearch — Motor de Busca Híbrido (BM25 + Busca Semântica Vetorial + RRF)
============================================================================
Desafio Integrador — Tendências em Ciência da Computação (UNIPÊ)
Prof. Me. Ricardo Roberto de Lima

Arquivo único em Streamlit que integra obrigatoriamente as 4 fases técnicas:

    Fase 1 — Ingestão do corpus médico e pré-processamento
             (tokenização, minúsculas, remoção de caracteres especiais e
              stopwords em português).
    Fase 2 — Motor léxico Okapi BM25 com parâmetros interativos:
             k1 (saturação de frequência) e b (normalização por comprimento).
    Fase 3 — Motor semântico vetorial: embeddings densos + similaridade de
             cosseno (captura de sinônimos / contexto clínico).
    Fase 4 — Fusão Reciprocal Rank Fusion (RRF):
             Score_RRF(D) = α · [1 / (k_rrf + Rank_BM25)]
                          + (1 − α) · [1 / (k_rrf + Rank_Semântico)]
             com k_rrf = 60 (constante de suavização de posição).
    Bônus  — Camada opcional de re-ranking com Cross-Encoder sobre o Top-3
             recuperado pela busca híbrida RRF.

Base de referência: app.py (estrutura/So caching Streamlit) e
vector_store.py (padrão SemanticStore com embeddings normalizados L2 e
similaridade de cosseno via produto interno).

Execução:
    streamlit run healthsearch.py

Dependências:
    streamlit, pandas, numpy, rank_bm25, sentence-transformers
    (todas com fallback documentado embutido, ver abaixo).
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

try:  # numpy acompanha pandas/sentence-transformers, mas protegemos mesmo assim
    import numpy as np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False

# ─────────────────────────────────────────────────────────────────────────
#  Dependências opcionais — com fallback documentado (Requisito Técnico)
# ─────────────────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi

    _HAS_RANK_BM25 = True
except Exception:  # pragma: no cover
    _HAS_RANK_BM25 = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    _HAS_ST = True
except Exception:  # pragma: no cover
    _HAS_ST = False


# ═════════════════════════════════════════════════════════════════════════
#  FASE 5 (Base de Dados) — Corpus médico hardcoded (6 diretrizes)
# ═════════════════════════════════════════════════════════════════════════
CORPUS: List[Dict[str, str]] = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "conteudo": (
            "Pacientes com dor precordial aguda e suspeita de síndrome "
            "coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até "
            "10 minutos."
        ),
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "conteudo": (
            "O uso imediato de ácido acetilsalicílico e antiagregantes "
            "plaquetários reduz a mortalidade no infarto agudo do miocárdio."
        ),
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "conteudo": (
            "A crise hipertensiva severa requer administração de "
            "anti-hipertensivos venosos e monitoramento contínuo da pressão "
            "arterial na UTI."
        ),
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "conteudo": (
            "O acidente vascular cerebral isquêmico agudo deve ser tratado "
            "com trombolíticos venosos em até quatro horas e meia do início "
            "dos sintomas."
        ),
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "conteudo": (
            "Parada cardiorrespiratória em adultos exige compressões "
            "torácicas contínuas de alta qualidade e desfibrilação precoce no "
            "código azul."
        ),
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "conteudo": (
            "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias "
            "complexas, recomenda-se a monitorização cardíaca contínua por "
            "telemetria."
        ),
    },
]


# ═════════════════════════════════════════════════════════════════════════
#  FASE 1 — Ingestão do Corpus e Pré-processamento
# ═════════════════════════════════════════════════════════════════════════
STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da",
    "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com",
    "sem", "sob", "sobre", "e", "ou", "que", "se", "ao", "aos", "the",
    "seu", "sua", "seus", "suas", "este", "esta", "isto", "esse", "essa",
    "isso", "aquele", "aquela", "aquilo", "como", "mais", "menos", "muito",
    "muita", "muitos", "muitas", "ser", "estar", "ter", "haver", "foi",
    "sao", "e", "ate", "tambem", "entre", "quando", "onde", "qual", "quais",
    "cada", "pelo", "pela", "pelos", "pelas", "num", "numa", "dele", "dela",
    "deles", "delas", "nao", "ja", "so", "mas", "porem", "ainda", "todo",
    "toda", "todos", "todas", "ne", "la", "aqui", "ali", "seja", "sendo",
    "deve", "devem", "ser", "requer", "exige",
}

_ACCENT_RE = re.compile(r"[̀-ͯ]")
_KEEP_RE = re.compile(r"[^a-z0-9\-\s]")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def strip_accents(text: str) -> str:
    """Remove diacríticos (NFKD) preservando a letra base."""
    return _ACCENT_RE.sub("", unicodedata.normalize("NFKD", text))


def preprocess(text: str) -> List[str]:
    """Fase 1 — pipeline de limpeza e tokenização.

    1. Converte para minúsculas.
    2. Remove acentos (unifica "coronária" ≈ "coronaria").
    3. Remove caracteres especiais, preservando dígitos e hífens
       (mantém códigos clínicos como CÓD-ECG-12D e dosagens 100mg).
    4. Tokeniza e elimina stopwords em português.
    5. Para tokens hifenizados também emite as subpartes, de modo que a
       consulta "ECG-12D" recupere o documento indexado como "CÓD-ECG-12D".
    """
    text = strip_accents(text.lower())
    text = _KEEP_RE.sub(" ", text)
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text):
        tok = raw.strip("-")
        if not tok or tok in STOPWORDS_PT:
            continue
        tokens.append(tok)
        if "-" in tok:
            for part in tok.split("-"):
                if part and part not in STOPWORDS_PT:
                    tokens.append(part)
    return tokens


# Documento indexável = título + conteúdo do trecho clínico
CORPUS_DOCS: List[str] = [f"{d['titulo']}. {d['conteudo']}" for d in CORPUS]
CORPUS_IDS: List[str] = [d["id"] for d in CORPUS]
CORPUS_TITLES: List[str] = [d["titulo"] for d in CORPUS]
CORPUS_TOKENS: List[List[str]] = [preprocess(doc) for doc in CORPUS_DOCS]


# ═════════════════════════════════════════════════════════════════════════
#  FASE 2 — Motor Léxico Okapi BM25
# ═════════════════════════════════════════════════════════════════════════
class BM25Manual:
    """Implementação didática do Okapi BM25 (fallback quando rank_bm25
    não está instalado). k1 e b são aplicados no momento da pontuação,
    portanto os sliders re-calculam a saturação e a normalização sem
    reconstruir o índice.

        IDF(t)   = ln(1 + (N − n(t) + 0.5) / (n(t) + 0.5))
        score(D) = Σ_t IDF(t) · [ f(t,D) · (k1 + 1) ]
                                 ─────────────────────────────────────────
                                 f(t,D) + k1 · (1 − b + b · |D| / avgdl)
    """

    def __init__(self, corpus_tokens: List[List[str]]) -> None:
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        for doc in corpus_tokens:
            freq: Dict[str, int] = {}
            for term in doc:
                freq[term] = freq.get(term, 0) + 1
            self.tf.append(freq)
            for term in freq:
                self.df[term] = self.df.get(term, 0) + 1

        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for term, n in self.df.items()
        }

    def get_scores(self, query_tokens: List[str], k1: float, b: float) -> List[float]:
        scores = [0.0] * self.N
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.N):
                f = self.tf[i].get(term, 0)
                if not f:
                    continue
                denom = f + k1 * (1.0 - b + b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (k1 + 1.0)) / denom
        return scores


@st.cache_resource(show_spinner=False)
def get_bm25_index(_tokens_key: Tuple[Tuple[str, ...], ...]):
    """Constrói o índice BM25 uma única vez (usa rank_bm25 se disponível)."""
    tokens = [list(t) for t in _tokens_key]
    if _HAS_RANK_BM25:
        return ("rank_bm25", tokens)
    return ("manual", BM25Manual(tokens))


def bm25_scores(query_tokens: List[str], k1: float, b: float) -> List[float]:
    """Fase 2 — pontuação BM25 com k1/b vindos dos sliders."""
    if not query_tokens:
        return [0.0] * len(CORPUS_TOKENS)

    kind, payload = get_bm25_index(tuple(tuple(t) for t in CORPUS_TOKENS))
    if kind == "rank_bm25":
        # rank_bm25 fixa k1/b no construtor -> recriamos (corpus minúsculo)
        model = BM25Okapi(payload, k1=k1, b=b)
        return [float(s) for s in model.get_scores(query_tokens)]
    return payload.get_scores(query_tokens, k1, b)


# ═════════════════════════════════════════════════════════════════════════
#  FASE 3 — Motor Semântico Vetorial
# ═════════════════════════════════════════════════════════════════════════
class SemanticEngine:
    """Mapeia documentos e consulta em um espaço vetorial e calcula a
    similaridade de cosseno.

    Modo primário  : sentence-transformers ("all-MiniLM-L6-v2"), embeddings
                     normalizados L2 -> produto interno = cosseno
                     (mesmo padrão do vector_store.SemanticStore).
    Modo fallback  : simulação vetorial documentada com TF-IDF + cosseno,
                     caso o pacote não esteja instalado. Não captura
                     sinônimos tão bem quanto os embeddings densos, servindo
                     apenas para manter a aplicação operacional.
    """

    def __init__(self, corpus_texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> None:
        self.corpus_texts = corpus_texts
        self.mode = "transformer" if _HAS_ST else "tfidf-sim"

        if self.mode == "transformer":
            self.model = SentenceTransformer(model_name)
            self.doc_emb = self._encode(corpus_texts)
        else:
            self._build_tfidf(corpus_texts)

    # ── modo transformer ───────────────────────────────────────────────
    def _encode(self, texts: List[str]):
        emb = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return emb.astype("float32")

    # ── modo fallback TF-IDF ───────────────────────────────────────────
    def _build_tfidf(self, texts: List[str]) -> None:
        docs_tokens = [preprocess(t) for t in texts]
        vocab = sorted({tok for doc in docs_tokens for tok in doc})
        self.vocab_index = {t: i for i, t in enumerate(vocab)}

        n_docs = len(texts)
        df = [0] * len(vocab)
        for doc in docs_tokens:
            for tok in set(doc):
                df[self.vocab_index[tok]] += 1
        self.idf = [math.log((n_docs + 1) / (d + 1)) + 1.0 for d in df]
        self.doc_vecs = [self._tfidf_vec(doc) for doc in docs_tokens]

    def _tfidf_vec(self, tokens: List[str]):
        vec = np.zeros(len(self.vocab_index), dtype="float32")
        for tok in tokens:
            j = self.vocab_index.get(tok)
            if j is not None:
                vec[j] += 1.0
        vec *= np.asarray(self.idf, dtype="float32")
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    # ── API pública ────────────────────────────────────────────────────
    def cosine_scores(self, query: str) -> List[float]:
        """Fase 3 — similaridade de cosseno consulta × cada documento."""
        if self.mode == "transformer":
            q = self._encode([query])[0]
            return [float(x) for x in (self.doc_emb @ q)]
        qv = self._tfidf_vec(preprocess(query))
        return [float(qv @ dv) for dv in self.doc_vecs]


@st.cache_resource(show_spinner="Carregando motor semântico (embeddings)…")
def get_semantic_engine() -> SemanticEngine:
    return SemanticEngine(CORPUS_DOCS)


@st.cache_resource(show_spinner="Carregando Cross-Encoder…")
def get_cross_encoder(name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(name)


# ═════════════════════════════════════════════════════════════════════════
#  FASE 4 — Reciprocal Rank Fusion (RRF)
# ═════════════════════════════════════════════════════════════════════════
K_RRF_SPEC = 60  # constante de suavização de posição (especificação)


def ranks_from_scores(scores: List[float]) -> List[int]:
    """Converte um vetor de scores em posições 1..N (1 = melhor).
    Empates são resolvidos pela ordem do índice do documento."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for position, doc_idx in enumerate(order, start=1):
        ranks[doc_idx] = position
    return ranks


def reciprocal_rank_fusion(
    bm25_score_list: List[float],
    semantic_score_list: List[float],
    alpha: float,
    k_rrf: int = K_RRF_SPEC,
) -> Tuple[List[float], List[int], List[int]]:
    """Fase 4 — aplica exatamente a fórmula do desafio:

        Score_RRF(D) = α · [1 / (k_rrf + Rank_BM25(D))]
                     + (1 − α) · [1 / (k_rrf + Rank_Semântico(D))]
    """
    rank_bm25 = ranks_from_scores(bm25_score_list)
    rank_sem = ranks_from_scores(semantic_score_list)
    fused = [
        alpha * (1.0 / (k_rrf + rank_bm25[i]))
        + (1.0 - alpha) * (1.0 / (k_rrf + rank_sem[i]))
        for i in range(len(bm25_score_list))
    ]
    return fused, rank_bm25, rank_sem


# ═════════════════════════════════════════════════════════════════════════
#  Helpers de apresentação
# ═════════════════════════════════════════════════════════════════════════
def ordered_results(scores: List[float]) -> List[Tuple[int, float]]:
    """(índice_doc, score) ordenado do melhor para o pior."""
    return sorted(
        enumerate(scores), key=lambda pair: (-pair[1], pair[0])
    )


def render_ranking(scores: List[float], top_k: int, score_label: str) -> None:
    """Lista visual de resultados com barra de progresso normalizada."""
    results = ordered_results(scores)[:top_k]
    max_score = max((s for _, s in results), default=0.0)
    min_score = min((s for _, s in results), default=0.0)
    span = (max_score - min_score) or 1.0

    for position, (doc_idx, score) in enumerate(results, start=1):
        with st.container(border=True):
            st.markdown(
                f"**#{position} — {CORPUS_IDS[doc_idx]} · {CORPUS_TITLES[doc_idx]}**  \n"
                f"{score_label}: `{score:.4f}`"
            )
            st.write(CORPUS[doc_idx]["conteudo"])
            norm = (score - min_score) / span
            st.progress(max(0.0, min(1.0, float(norm))))


# ═════════════════════════════════════════════════════════════════════════
#  APLICAÇÃO STREAMLIT
# ═════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HealthSearch — Busca Híbrida BM25 + Semântica + RRF",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 HealthSearch — Motor de Busca Híbrido")
st.caption(
    "Okapi BM25 (léxico) · Busca Semântica Vetorial (embeddings) · "
    "Reciprocal Rank Fusion (RRF)"
)

# ── Sidebar: calibração das 4 fases ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Calibração")

    st.subheader("Fase 2 · BM25")
    k1 = st.slider(
        "k₁ — saturação de frequência de termo",
        min_value=0.0, max_value=3.0, value=1.2, step=0.1,
        help="Controla o quão rápido a repetição de um termo satura o score.",
    )
    b = st.slider(
        "b — normalização pelo comprimento do documento",
        min_value=0.0, max_value=1.0, value=0.75, step=0.05,
        help="0 = ignora o tamanho do documento; 1 = normaliza totalmente.",
    )

    st.subheader("Fase 4 · Fusão RRF")
    alpha = st.slider(
        "α — peso balanceador (BM25 × Semântico)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="α·BM25 + (1−α)·Semântico. α=1 só léxico, α=0 só semântico.",
    )
    k_rrf = st.slider(
        "k_RRF — constante de suavização de posição",
        min_value=10, max_value=100, value=K_RRF_SPEC, step=1,
        help="Especificação do desafio: k_RRF = 60.",
    )

    top_k = st.number_input(
        "Top-K exibido", min_value=1, max_value=len(CORPUS), value=len(CORPUS), step=1
    )

    st.divider()
    use_cross_encoder = st.checkbox(
        "🎯 Bônus — Re-ranking Cross-Encoder (Top-3 do RRF)",
        value=False,
        help="cross-encoder/ms-marco-MiniLM-L-6-v2 sobre os 3 melhores do híbrido.",
    )

    st.divider()
    st.caption("**Back-ends ativos**")
    st.caption(
        f"BM25: `{'rank_bm25' if _HAS_RANK_BM25 else 'implementação própria'}`"
    )
    st.caption(
        "Semântico: "
        f"`{'sentence-transformers' if _HAS_ST else 'simulação TF-IDF (fallback)'}`"
    )
    if not _HAS_ST:
        st.warning(
            "sentence-transformers não instalado — usando simulação vetorial "
            "TF-IDF documentada. Instale para embeddings densos reais.",
            icon="⚠️",
        )

# ── Motores ────────────────────────────────────────────────────────────
semantic_engine = get_semantic_engine()

# ── Consultas de exemplo (demonstram os pontos cegos) ──────────────────
if "hs_query" not in st.session_state:
    st.session_state.hs_query = ""

st.markdown("**Consultas de exemplo:**")
EXAMPLES = [
    "ataque cardíaco",
    "infarto",
    "ECG-12D",
    "AAS ácido acetilsalicílico",
    "derrame cerebral",
    "parada cardíaca",
]
ex_cols = st.columns(len(EXAMPLES))
for col, example in zip(ex_cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.hs_query = example
        st.rerun()

query = st.text_input(
    "🔎 Consulta clínica",
    key="hs_query",
    placeholder="Ex.: síndrome coronariana aguda, ECG-12D, AAS 100mg, AVC…",
)

if not query.strip():
    st.info(
        "Digite uma consulta (ou use um exemplo acima) para comparar os três "
        "motores de busca lado a lado."
    )
    st.stop()

# ── Pipeline de busca ─────────────────────────────────────────────────
query_tokens = preprocess(query)
bm_scores = bm25_scores(query_tokens, k1, b)
sem_scores = semantic_engine.cosine_scores(query)
fused_scores, rank_bm25, rank_sem = reciprocal_rank_fusion(
    bm_scores, sem_scores, alpha, k_rrf
)
rank_rrf = ranks_from_scores(fused_scores)

st.caption(
    "Tokens da consulta após pré-processamento (Fase 1): "
    + (", ".join(f"`{t}`" for t in query_tokens) or "_(nenhum token válido)_")
)

# ── Diagnóstico dos pontos cegos ─────────────────────────────────────
lexical_blind = max(bm_scores) <= 0.0
if lexical_blind:
    st.warning(
        "🔍 **Ponto cego léxico detectado:** o BM25 não encontrou nenhuma "
        "correspondência exata de termo. A Busca Semântica (sinônimos "
        "médicos) e o RRF cobrem essa lacuna.",
        icon="🧠",
    )
code_like = any(re.search(r"\d", t) or "-" in t for t in query_tokens)
if code_like and not lexical_blind:
    st.info(
        "🔢 **Consulta com código/dosagem:** aqui o BM25 tende a ser mais "
        "preciso que o semântico puro — o RRF preserva essa precisão.",
        icon="📘",
    )

# ── Abas comparativas ────────────────────────────────────────────────
tab_lex, tab_sem, tab_rrf, tab_matrix = st.tabs(
    ["📘 Léxico (BM25)", "🧠 Semântico", "⚗️ Híbrido RRF", "📊 Matriz Comparativa"]
)

with tab_lex:
    st.subheader("Fase 2 — Okapi BM25")
    st.caption(f"Parâmetros ativos: k₁ = {k1:.2f} · b = {b:.2f}")
    render_ranking(bm_scores, top_k, "Score BM25")

with tab_sem:
    st.subheader("Fase 3 — Busca Semântica Vetorial")
    st.caption(
        f"Similaridade de cosseno · modo: {semantic_engine.mode} "
        f"({'all-MiniLM-L6-v2' if semantic_engine.mode == 'transformer' else 'TF-IDF simulado'})"
    )
    render_ranking(sem_scores, top_k, "Cosseno")

with tab_rrf:
    st.subheader("Fase 4 — Reciprocal Rank Fusion")
    st.latex(
        r"Score_{RRF}(D) = \alpha \cdot \frac{1}{k_{RRF} + Rank_{BM25}(D)}"
        r" + (1-\alpha) \cdot \frac{1}{k_{RRF} + Rank_{Sem}(D)}"
    )
    st.caption(f"α = {alpha:.2f} · k_RRF = {k_rrf}")
    render_ranking(fused_scores, top_k, "Score RRF")

    if use_cross_encoder:
        st.divider()
        st.subheader("🎯 Bônus — Re-ranking com Cross-Encoder")
        if not _HAS_ST:
            st.error(
                "sentence-transformers indisponível — não é possível aplicar "
                "o Cross-Encoder."
            )
        else:
            top3_idx = [idx for idx, _ in ordered_results(fused_scores)[:3]]
            cross_encoder = get_cross_encoder()
            pairs = [[query, CORPUS_DOCS[i]] for i in top3_idx]
            ce_scores = [float(s) for s in cross_encoder.predict(pairs)]

            reranked = sorted(
                zip(top3_idx, ce_scores), key=lambda p: -p[1]
            )
            ce_rows = []
            for new_pos, (doc_idx, ce_score) in enumerate(reranked, start=1):
                old_pos = top3_idx.index(doc_idx) + 1
                ce_rows.append(
                    {
                        "Documento": CORPUS_IDS[doc_idx],
                        "Título": CORPUS_TITLES[doc_idx],
                        "Rank RRF": old_pos,
                        "Rank Cross-Encoder": new_pos,
                        "Δ posição": old_pos - new_pos,
                        "Score RRF": round(fused_scores[doc_idx], 5),
                        "Score Cross-Encoder": round(ce_score, 4),
                    }
                )
            st.dataframe(
                pd.DataFrame(ce_rows), use_container_width=True, hide_index=True
            )
            st.caption(
                "O Cross-Encoder lê consulta + documento juntos, produzindo "
                "uma nota de relevância mais fina sobre os 3 finalistas."
            )

with tab_matrix:
    st.subheader("Matriz Comparativa de Rankings")

    matrix_rows = []
    for i in range(len(CORPUS)):
        matrix_rows.append(
            {
                "Documento": CORPUS_IDS[i],
                "Título": CORPUS_TITLES[i],
                "Score BM25": round(bm_scores[i], 4),
                "Rank BM25": rank_bm25[i],
                "Cosseno": round(sem_scores[i], 4),
                "Rank Semântico": rank_sem[i],
                "Score RRF": round(fused_scores[i], 5),
                "Rank RRF": rank_rrf[i],
            }
        )
    matrix_df = pd.DataFrame(matrix_rows).sort_values("Rank RRF").reset_index(drop=True)
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)

    st.markdown("**Comparação visual de posições (menor = melhor)**")
    chart_df = (
        pd.DataFrame(
            {
                "BM25": rank_bm25,
                "Semântico": rank_sem,
                "RRF": rank_rrf,
            },
            index=CORPUS_IDS,
        )
    )
    st.bar_chart(chart_df)

    # Métricas de desempenho da busca
    st.markdown("**Métricas de desempenho**")
    m1, m2, m3, m4 = st.columns(4)
    best_bm = min(range(len(CORPUS)), key=lambda i: rank_bm25[i])
    best_sem = min(range(len(CORPUS)), key=lambda i: rank_sem[i])
    best_rrf = min(range(len(CORPUS)), key=lambda i: rank_rrf[i])
    agreement = sum(
        1 for i in range(len(CORPUS)) if rank_bm25[i] == rank_sem[i]
    )
    m1.metric("Top-1 BM25", CORPUS_IDS[best_bm])
    m2.metric("Top-1 Semântico", CORPUS_IDS[best_sem])
    m3.metric("Top-1 Híbrido RRF", CORPUS_IDS[best_rrf])
    m4.metric("Docs em mesma posição", f"{agreement}/{len(CORPUS)}")

    rho = chart_df[["BM25", "Semântico"]].corr(method="spearman").iloc[0, 1]
    st.caption(
        f"Correlação de Spearman entre os rankings BM25 e Semântico: "
        f"**{rho:.2f}** — quanto menor, mais os motores discordam e maior o "
        f"valor agregado da fusão RRF."
    )

st.divider()
with st.expander("📋 Corpus médico carregado (6 diretrizes)"):
    for d in CORPUS:
        st.markdown(f"**{d['id']} · {d['titulo']}**")
        st.caption(d["conteudo"])
