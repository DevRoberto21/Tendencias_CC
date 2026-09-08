import streamlit as st
from vector_store import SemanticStore

# ────────────────────────────────────────────────
#  Configuração da página
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Busca Semântica — FAISS",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Busca Semântica com FAISS")
st.caption(
    "Ingestão e consulta de dados semânticos usando embeddings e busca vetorial."
)


# ────────────────────────────────────────────────
#  Cache do SemanticStore
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo de embeddings…")
def get_store() -> SemanticStore:
    return SemanticStore(
        model_name="all-MiniLM-L6-v2",
        index_dir="faiss_index",
    )


store = get_store()


# ────────────────────────────────────────────────
#  Barra lateral — Ingestão de novos documentos
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("📥 Ingestão de Dados")

    # ── Opção 1: Digitar textos ──
    st.subheader("Adicionar texto manualmente")
    new_text = st.text_area(
        "Cole ou digite um ou mais parágrafos (um documento por linha vazia):",
        height=150,
        placeholder="Exemplo:\nPython é uma linguagem de programação.\nFAISS é uma biblioteca de busca vetorial.",
    )

    if st.button("Adicionar textos", use_container_width=True):
        if new_text.strip():
            raw_chunks = new_text.strip().split("\n\n")
            docs = [c.strip() for c in raw_chunks if c.strip()]

            if docs:
                store.add_documents(
                    docs, extra_meta=[{"source": "manual"}] * len(docs)
                )
                st.success(f"✅ {len(docs)} documento(s) adicionado(s)!")
                st.rerun()
        else:
            st.warning("Nenhum texto informado.")

    st.divider()

    # ── Opção 2: Upload de arquivo .txt ──
    st.subheader("Upload de arquivo .txt")
    uploaded = st.file_uploader(
        "Selecione um arquivo de texto", type=["txt"], key="txt_uploader"
    )

    if uploaded and st.button("Ingerir arquivo", use_container_width=True):
        try:
            content = uploaded.read().decode("utf-8")
            docs = [
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ]

            if docs:
                store.add_documents(
                    docs, extra_meta=[{"source": uploaded.name}] * len(docs)
                )
                st.success(f"✅ {len(docs)} linha(s) ingerida(s) do arquivo!")
                st.rerun()
            else:
                st.warning("O arquivo está vazio.")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")

    st.divider()

    # ── Estatísticas ──
    st.subheader("📊 Estatísticas")
    st.metric("Documentos no banco", store.total_documents)

    if st.button("🗑️ Limpar banco", use_container_width=True):
        store.clear()
        st.success("Banco de dados limpo!")
        st.rerun()


# ────────────────────────────────────────────────
#  Área principal — Consulta semântica
# ────────────────────────────────────────────────
st.header("🔎 Consulta Semântica")

col_q, col_k = st.columns([4, 1])
with col_q:
    query = st.text_input(
        "Pergunte algo:",
        placeholder="Ex.: O que é um banco de dados vetorial?",
    )
with col_k:
    top_k = st.number_input(
        "Top-K", min_value=1, max_value=50, value=5, step=1
    )

if query.strip():
    if store.total_documents == 0:
        st.info(
            "O banco está vazio. Adicione documentos na barra lateral para realizar buscas."
        )
    else:
        with st.spinner("Buscando…"):
            results = store.search(query, top_k=top_k)

        if results:
            st.subheader(f"Resultados (top {len(results)})")
            for rank, doc in enumerate(results, start=1):
                score = doc["score"]
                source = doc.get("source", "—")

                if score >= 0.7:
                    color = "green"
                elif score >= 0.4:
                    color = "orange"
                else:
                    color = "red"

                with st.container(border=True):
                    st.markdown(
                        f"**#{rank}** &nbsp; | &nbsp; "
                        f"Similaridade: :{color}[**{score:.4f}**] &nbsp; | &nbsp; "
                        f"Origem: `{source}`"
                    )
                    st.write(doc["text"])

                    # Garante valor entre 0.0 e 1.0 sem precisar do numpy
                    progress_val = max(0.0, min(float(score), 1.0))
                    st.progress(progress_val)
        else:
            st.info("Nenhum resultado relevante encontrado.")
else:
    st.info("👆 Digite uma consulta acima para buscar no banco vetorial.")


# ────────────────────────────────────────────────
#  Visualizar todos os documentos armazenados
# ────────────────────────────────────────────────
st.divider()
with st.expander("📋 Ver todos os documentos armazenados"):
    all_docs = store.get_all_documents()
    if all_docs:
        for d in all_docs:
            st.markdown(f"**ID {d['id']}** · `{d.get('source', '—')}`")
            st.caption(d["text"])
            st.divider()
    else:
        st.write("O banco está vazio.")