# HealthSearch — Motor de Busca Híbrido (BM25 + Semântica Vetorial + RRF)

Desafio Integrador — **Tendências em Ciência da Computação** (UNIPÊ)
Recuperação de Informação / Processamento de Linguagem Natural
Prof. Me. Ricardo Roberto de Lima

Protótipo em **Streamlit** que combina a precisão léxica do **Okapi BM25** com a
inteligência contextual da **Busca Semântica Vetorial (embeddings)** através do
**Reciprocal Rank Fusion (RRF)**, aplicado a um corpus de 6 diretrizes médicas.

O objetivo é demonstrar visualmente como a busca híbrida resolve os pontos cegos
de cada abordagem:

- **BM25 puro** falha com sinônimos: buscar *"ataque cardíaco"* não recupera
  diretrizes escritas como *"infarto agudo do miocárdio"* ou *"síndrome
  coronariana"*.
- **Semântico puro** perde precisão com códigos e dosagens exatas
  (ex.: `CÓD-ECG-12D`, `AAS 100mg`), trazendo trechos genéricos.

---

## 1. Requisitos

- **Python 3.9+** (testado em 3.11)
- ~200 MB livres em disco (o modelo de embeddings é baixado na 1ª execução)
- Conexão com a internet na primeira execução (download do modelo)

Bibliotecas (em `requirements.txt`):

| Biblioteca | Uso |
|---|---|
| `streamlit` | Interface web interativa |
| `pandas` | Tabelas e matriz comparativa |
| `numpy` | Operações vetoriais |
| `rank_bm25` | Algoritmo Okapi BM25 |
| `sentence-transformers` | Embeddings densos + Cross-Encoder (bônus) |

> **Fallback embutido:** se `rank_bm25` ou `sentence-transformers` não estiverem
> instalados, o app ainda funciona — usa uma implementação própria de BM25 e uma
> simulação vetorial documentada (TF-IDF + cosseno). Um aviso é exibido na
> barra lateral. Para o resultado real do desafio, instale todas as dependências.

---

## 2. Instalação

```bash
# 1. Entre na pasta do projeto
cd "Desafio"

# 2. (Recomendado) crie e ative um ambiente virtual
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## 3. Como executar

```bash
streamlit run healthsearch.py
```

O navegador abre automaticamente em `http://localhost:8501`.

> **Primeira execução:** o `sentence-transformers` baixa o modelo
> `all-MiniLM-L6-v2` (~90 MB). Pode levar 1–2 minutos. Nas próximas vezes ele
> carrega do cache local.

Para encerrar: `Ctrl + C` no terminal.

---

## 4. Como usar a interface

### 4.1 Fazer uma busca

1. Digite uma consulta clínica no campo **🔎 Consulta clínica** — ou clique em
   um dos **botões de exemplo** (`ataque cardíaco`, `infarto`, `ECG-12D`,
   `AAS ácido acetilsalicílico`, `derrame cerebral`, `parada cardíaca`).
2. Os resultados aparecem imediatamente nas 4 abas e são recalculados a cada
   ajuste de parâmetro.

### 4.2 As 4 abas

| Aba | Fase | O que mostra |
|---|---|---|
| 📘 **Léxico (BM25)** | Fase 2 | Ranking por correspondência exata de termos, com os `k₁` e `b` atuais |
| 🧠 **Semântico** | Fase 3 | Ranking por similaridade de cosseno dos embeddings (captura sinônimos) |
| ⚗️ **Híbrido RRF** | Fase 4 | Ranking fundido pela fórmula RRF; opção de re-ranking Cross-Encoder (bônus) |
| 📊 **Matriz Comparativa** | — | Tabela com score/rank de cada motor, gráfico de barras das posições, métricas e correlação de Spearman |

### 4.3 Controles da barra lateral

| Controle | Faixa (padrão) | Efeito |
|---|---|---|
| **`k₁`** — saturação de frequência | 0.0 – 3.0 (**1.2**) | Quão rápido a repetição de um termo deixa de aumentar o score. Baixo = 2ª ocorrência quase não conta; alto = repetição continua pesando. |
| **`b`** — normalização por comprimento | 0.0 – 1.0 (**0.75**) | Penaliza documentos longos. `b=0` ignora o tamanho; `b=1` normaliza totalmente. |
| **`α`** — peso balanceador RRF | 0.0 – 1.0 (**0.5**) | `α=1` → só BM25; `α=0` → só semântico; `0.5` → equilíbrio. |
| **`k_RRF`** — suavização de posição | 10 – 100 (**60**) | Constante da fórmula RRF. Manter em **60** (especificação). |
| **Top-K** | 1 – 6 (**6**) | Quantidade de resultados listados. |
| **☑ Re-ranking Cross-Encoder** | desligado | Bônus: reavalia o Top-3 do RRF com `cross-encoder/ms-marco-MiniLM-L-6-v2` e mostra tabela antes/depois. |

### 4.4 Fórmula RRF aplicada

```
Score_RRF(D) = α · [ 1 / (k_RRF + Rank_BM25(D)) ]
             + (1 − α) · [ 1 / (k_RRF + Rank_Semântico(D)) ]

com k_RRF = 60
```

---

## 5. Roteiro de demonstração (pontos cegos)

| Consulta | Comportamento observado | Conclusão |
|---|---|---|
| `ataque cardíaco` | BM25 não pontua nada (nenhum documento usa o termo) → aviso de **ponto cego léxico**. O motor semântico recupera o Doc 2 (*infarto agudo do miocárdio*). | Sinônimos só são resolvidos pelo semântico / RRF. |
| `ECG-12D` ou `AAS 100mg` | BM25 acerta em cheio os Doc 1 e Doc 6; o semântico puro "dilui" em documentos cardíacos genéricos. | Precisão de códigos/dosagens vem do BM25 / RRF. |
| Qualquer consulta com `α ≈ 0.5` | O RRF combina os dois rankings, equilibrando precisão e cobertura. | A busca híbrida cobre os dois pontos cegos. |

Use a aba **📊 Matriz Comparativa** para gerar o **gráfico de comparação de
ranks** exigido no relatório técnico (PDF).

---

## 6. Estrutura do projeto

```
Desafio/
├── healthsearch.py       # Aplicação completa (as 4 fases + bônus) — arquivo único
├── requirements.txt      # Dependências
├── README.md             # Este arquivo
├── app.py                # Base de referência: estrutura Streamlit / caching
└── vector_store.py       # Base de referência: padrão SemanticStore (embeddings + cosseno)
```

### Fases implementadas em `healthsearch.py`

| Fase | Onde | Descrição |
|---|---|---|
| **1. Pré-processamento** | `preprocess()` | Minúsculas, remoção de acentos, limpeza de caracteres especiais (preserva dígitos e hífens de códigos), tokenização e stopwords PT-BR. |
| **2. BM25** | `BM25Manual` / `bm25_scores()` | Okapi BM25 com `k₁` e `b` aplicados na pontuação (sliders recalculam sem reindexar). Usa `rank_bm25` quando disponível. |
| **3. Semântico** | `SemanticEngine` | Embeddings `all-MiniLM-L6-v2` normalizados L2, similaridade de cosseno. Fallback TF-IDF documentado. |
| **4. RRF** | `reciprocal_rank_fusion()` | Fusão exata pela fórmula do desafio, com `α` e `k_RRF = 60`. |
| **Bônus** | `get_cross_encoder()` | Re-ranking Cross-Encoder sobre o Top-3 híbrido, com comparação da nota de relevância. |

---

## 7. Solução de problemas

| Sintoma | Causa provável | Solução |
|---|---|---|
| `ModuleNotFoundError: streamlit` | Dependências não instaladas ou venv não ativado | `pip install -r requirements.txt` |
| Aviso amarelo "usando simulação TF-IDF" | `sentence-transformers` ausente | `pip install sentence-transformers` |
| 1ª execução muito lenta / trava | Download do modelo de embeddings | Aguardar; garantir conexão com a internet |
| `streamlit: command not found` | venv não ativado | Ativar o venv ou usar `python -m streamlit run healthsearch.py` |
| Porta 8501 ocupada | Outra instância rodando | `streamlit run healthsearch.py --server.port 8502` |

---

## 8. Entrega

O enunciado pede o arquivo com o nome `healthsearch_app.py`. Para gerar a cópia
final:

```bash
cp healthsearch.py healthsearch_app.py
```

Além do código, entregar o **relatório técnico em PDF (máx. 2 páginas)** com a
arquitetura da solução, o gráfico de comparação de ranks (aba Matriz
Comparativa) e a divisão de tarefas da equipe.
