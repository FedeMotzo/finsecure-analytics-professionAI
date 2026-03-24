"""
Pipeline di ingestion per FinSecure Analytics.

Strategia "Structural Chunking":
  LlamaParse estrae il PDF in Markdown preservando tabelle e struttura.
  MarkdownElementNodeParser usa un LLM per produrre nodi distinti per tabelle
  e testo narrativo, evitando che una tabella finanziaria venga spezzata a metà.
"""

import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Costanti directories
ROOT_DIR = Path(__file__).parent.parent
MOCK_DATA_DIR = ROOT_DIR / "mock_data"
CHROMA_PERSIST_DIR = str(ROOT_DIR / "chroma_db_storage")
CHROMA_COLLECTION_NAME = "finsecure_docs"
DB_PATH = ROOT_DIR / "audit.db"


def load_environment() -> None:
    """Carica le variabili d'ambiente da .env e verifica le chiavi obbligatorie.
    """
    env_path = ROOT_DIR / ".env"
    load_dotenv(dotenv_path=env_path)

    missing = [k for k in ("LLAMA_CLOUD_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        log.error("Variabili d'ambiente mancanti nel file .env: %s", ", ".join(missing))
        sys.exit(1)

    log.info("Variabili d'ambiente caricate da: %s", env_path)


def parse_pdfs_with_llamaparse() -> list[Any]:
    """Esegue il parsing dei PDF con LlamaParse.

    LlamaParse preserva la struttura delle tabelle finanziarie estraendole
    in Markdown, rendendo il chunking strutturale più affidabile.

    Returns:
        Lista di oggetti Document.
    """
    from llama_parse import LlamaParse

    pdf_files = sorted(MOCK_DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"Nessun file PDF trovato in: {MOCK_DATA_DIR}")

    log.info("PDF trovati: %d", len(pdf_files))
    for p in pdf_files:
        log.info("  → %s  (%.1f KB)", p.name, p.stat().st_size / 1024)

    parser = LlamaParse(
        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
        result_type="markdown",
        verbose=False,
    )

    all_documents: list[Any] = []
    for pdf_path in pdf_files:
        log.info("Parsing: %s ...", pdf_path.name)
        try:
            docs = parser.load_data(str(pdf_path))
            log.info("  Pagine/doc estratti: %d", len(docs))
            all_documents.extend(docs)
        except Exception as exc:
            log.error("  ERRORE nel parsing di '%s': %s", pdf_path.name, exc)
            raise

    log.info("Totale documenti LlamaParse: %d", len(all_documents))
    return all_documents


def build_structural_nodes(documents: list[Any]) -> list[Any]:
    """Applica il chunking strutturale con MarkdownElementNodeParser.

    Usa un llm per identificare e separare tabelle dal testo narrativo.
    Il risultato sono nodi distinti: le tabelle non vengono mai spezzate.

    Args:
        documents: Lista di Document ottenuti da LlamaParse.

    Returns:
        Lista piatta di BaseNode.
    """
    from llama_index.core.node_parser import MarkdownElementNodeParser
    from llama_index.core.schema import IndexNode
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    node_parser = MarkdownElementNodeParser(
        llm=llm,
        num_workers=4,
    )

    log.info("Avvio MarkdownElementNodeParser...")
    try:
        result = node_parser.get_nodes_from_documents(documents, show_progress=True)
    except Exception as exc:
        log.error("Errore durante il node parsing: %s", exc)
        raise

    # L'API può restituire una tupla (base_nodes, objects) o una lista piatta
    if isinstance(result, tuple):
        base_nodes, objects = result
        all_nodes: list[Any] = base_nodes + objects
        log.info("Nodi base: %d  |  Index nodes (tabelle): %d", len(base_nodes), len(objects))
    else:
        all_nodes = result

    # Statistiche: distinzione tabelle vs testo
    table_nodes = [
        n for n in all_nodes
        if isinstance(n, IndexNode)
        or (hasattr(n, "metadata") and str(n.metadata.get("type", "")).lower() == "table")
    ]
    text_nodes = [n for n in all_nodes if n not in table_nodes]

    log.info("─" * 50)
    log.info("Nodi TABELLA  : %3d", len(table_nodes))
    log.info("Nodi TESTO    : %3d", len(text_nodes))
    log.info("Totale nodi   : %3d", len(all_nodes))
    log.info("─" * 50)

    return all_nodes


def build_chroma_index(nodes: list[Any]) -> Any:
    """Crea il VectorStoreIndex in ChromaDB.

    Args:
        nodes: Lista di BaseNode da indicizzare.

    Returns:
        Istanza VectorStoreIndex pronta per le query.
    """
    import chromadb
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    # Embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.embed_model = embed_model

    # ChromaDB
    log.info("Inizializzazione ChromaDB in: %s", CHROMA_PERSIST_DIR)
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        log.info("Collection ChromaDB: '%s'", CHROMA_COLLECTION_NAME)
    except Exception as exc:
        log.error("Errore nell'inizializzazione di ChromaDB: %s", exc)
        raise

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Indicizzazione
    log.info("Avvio indicizzazione di %d nodi in ChromaDB ...", len(nodes))
    try:
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
    except Exception as exc:
        log.error("Errore durante l'indicizzazione: %s", exc)
        raise

    log.info("Indicizzazione completata.")
    return index


KPI_EXTRACTION_PROMPT = """
Sei un analista finanziario. Analizza tutti i documenti disponibili e restituisci
ESCLUSIVAMENTE un oggetto JSON valido. Nessun testo aggiuntivo prima o dopo.

Regole di formato:
- Tutti i valori devono essere numeri interi puri (niente simboli K, M o €)
- I valori in Milioni vanno convertiti: 1.2M → 1200, 3.5M → 3500
- Se un valore non è rilevabile con certezza, usa null

{
  "ricavi_totali":      <Ricavi Totali o Total Revenue del periodo (es. fatturato lordo), intero in migliaia di EUR>,
  "ebitda_dichiarato":  <EBITDA come esplicitamente dichiarato nel conto economico Q3, intero in migliaia>,
  "ebit":               <EBIT = Utile Operativo prima di interessi e imposte, intero in migliaia>,
  "da":                 <Ammortamenti e Svalutazioni (D&A / Depreciation & Amortization), intero in migliaia>,
  "ebitda_ricalcolato": <Calcola tu: EBIT + D&A, intero in migliaia>,
  "total_debt_market":  <Debito Totale (Total Debt) citato nell'Analisi di Rischio di Mercato, intero in migliaia>
}
""".strip()


def init_db() -> None:
    """Crea le tabelle SQLite se non esistono.

    Schema:
        kpis     — KPI estratti durante l'ingestion
        findings — Criticità rilevate dall'agente durante la chat
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS kpis (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                label        TEXT    NOT NULL,
                value        TEXT    NOT NULL,
                status       TEXT    NOT NULL DEFAULT 'ok',
                source_doc   TEXT,
                extracted_at TEXT    DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS findings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                severity    TEXT    NOT NULL,
                description TEXT    NOT NULL,
                source_docs TEXT,
                found_at    TEXT    DEFAULT (datetime('now','localtime'))
            );
        """)
    log.info("DB SQLite inizializzato: %s", DB_PATH)


def extract_and_save_kpis(index: Any) -> dict:
    """Interroga il VectorStoreIndex per estrarre i KPI chiave e li salva in SQLite.

    Args:
        index: VectorStoreIndex già popolato con i documenti.

    Returns:
        Dizionario con i KPI estratti.
    """
    import json

    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI

    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        response_mode="tree_summarize",
    )

    log.info("Estrazione KPI dai documenti indicizzati ...")
    try:
        raw = str(query_engine.query(KPI_EXTRACTION_PROMPT))
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        kpis_raw: dict = json.loads(raw[start:end])
        # Sostituisce None (da JSON null) e stringhe vuote con "N/D
        kpis: dict = {k: (str(v) if v else "N/D") for k, v in kpis_raw.items()}
    except Exception as exc:
        log.warning("Estrazione KPI fallita (%s) — uso placeholder.", exc)
        kpis = {k: "N/D" for k in (
            "ricavi_totali", "ebitda_dichiarato", "ebit", "da",
            "ebitda_ricalcolato", "total_debt_market",
        )}

    # Determina lo status EBITDA confrontando dichiarato vs ricalcolato
    ebitda_status = (
        "critical"
        if kpis.get("ebitda_dichiarato", "N/D") != "N/D"
        and kpis.get("ebitda_ricalcolato", "N/D") != "N/D"
        and kpis["ebitda_dichiarato"] != kpis["ebitda_ricalcolato"]
        else "ok"
    )

    def _v(key: str) -> str:
        """Restituisce il valore del KPI o 'N/D' se assente/None/vuoto."""
        return kpis.get(key) or "N/D"

    rows = [
        ("Ricavi Totali Q3",    _v("ricavi_totali"),      "ok",          "report_Q3.pdf"),
        ("EBITDA Dichiarato",   _v("ebitda_dichiarato"),  ebitda_status, "report_Q3.pdf"),
        ("EBITDA Ricalcolato",  _v("ebitda_ricalcolato"), ebitda_status, "report_Q3.pdf"),
        ("EBIT",                _v("ebit"),               "ok",          "report_Q3.pdf"),
        ("Total Debt (Market)", _v("total_debt_market"),  "warning",     "market_exposure.pdf"),
    ]

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM kpis")          # re-ingestion idempotente
        conn.executemany(
            "INSERT INTO kpis (label, value, status, source_doc) VALUES (?,?,?,?)",
            rows,
        )

    log.info("KPI salvati in SQLite (%d righe):", len(rows))
    for label, value, status, _ in rows:
        log.info("  %-25s %-12s [%s]", label, value, status)

    return kpis


def main() -> None:
    """Entry point principale della pipeline di ingestion."""
    log.info("=" * 55)
    log.info("  FinSecure Analytics — RAG Ingestion Pipeline")
    log.info("=" * 55)

    load_environment()

    # Step 1: DB SQLite (crea tabelle se assenti)
    init_db()

    # Step 2: Parsing PDF → Markdown
    documents = parse_pdfs_with_llamaparse()

    # Step 3: Structural chunking → nodi separati per tabelle e testo
    nodes = build_structural_nodes(documents)

    # Step 4: Embedding + ChromaDB
    index = build_chroma_index(nodes)

    # Step 5: Estrazione KPI → SQLite
    extract_and_save_kpis(index)

    log.info("=" * 55)
    log.info("Pipeline completata.")
    log.info("Vector DB : %s", CHROMA_PERSIST_DIR)
    log.info("SQLite DB : %s", DB_PATH)
    log.info("Nodi      : %d", len(nodes))
    log.info("=" * 55)

    return index


if __name__ == "__main__":
    main()
