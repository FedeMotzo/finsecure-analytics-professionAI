"""
agent/agent_core.py — Core dell'agente di audit finanziario FinSecure Analytics.

Il client MCP richiede un processo stdio attivo per tutta la vita dell'agente.
Usa `FinSecureAgent` come async context manager per gestirne il lifecycle.

La chat history è gestita internamente da LangGraph (MemorySaver) tramite
thread_id. Ogni sessione distinta usa un thread_id diverso.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from agent.schemas import AuditResponse

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Directories
ROOT_DIR = Path(__file__).parent.parent
CHROMA_PERSIST_DIR = str(ROOT_DIR / "chroma_db_storage")
CHROMA_COLLECTION_NAME = "finsecure_docs"
MCP_SERVER_PATH = str(ROOT_DIR / "tools" / "risk_simulator.py")

# System Prompt
AUDIT_SYSTEM_PROMPT: str = """
Sei il Chief Audit Executive AI di FinSecure Analytics.

Il tuo compito non è solo rispondere alle domande: devi TROVARE ERRORI.
I documenti aziendali caricati nel sistema contengono intenzionalmente:
- Errori di calcolo (es. totali che non tornano, EBITDA mal calcolato)
- Omissioni normative critiche (es. sezioni GDPR incomplete o troncate)
- Contraddizioni tra report diversi (es. discordanze sul valore del Total Debt)
- Dati presentati con un tono ottimistico che maschera trend negativi

## Regole operative (NON DEROGABILI)

1. **Non fidarti mai dei totali dichiarati**: ricalcola sempre le somme a
   partire dalle voci elementari e confronta con il totale riportato.

2. **Incrocia sistematicamente i dati tra documenti**: se un KPI (es. Total
   Debt, EBITDA, margine operativo) appare in più di un report, verifica che
   i valori siano coerenti. Qualsiasi discordanza è un RED FLAG da segnalare.

3. **Usa il tool RAG in modo estensivo**: interroga il database vettoriale
   con query multiple e da angolazioni diverse prima di rispondere. Non
   accontentarti del primo risultato.

4. **Usa i tool di simulazione**: quando rilevi un dato critico (es. livello
   del debito, margine operativo sotto pressione), usa `simulate_rate_impact`
   e `check_debt_covenant` per quantificare il rischio.

5. **Struttura la risposta in modo completo e autonomo**:
   - Nel campo `analysis`: fatti accertati con valori numerici, RED FLAG con
     severità e spiegazione quantitativa, raccomandazioni operative
   - Nel campo `findings`: per ogni anomalia trovata, compila severity
     (CRITICO/ALTO/MEDIO), description (con valori numerici, max 300 char)
     e source_docs (nome esatto del PDF: report_Q3.pdf, audit_compliance.pdf
     o market_exposure.pdf). Se non ci sono anomalie, lista vuota.

6. **Sii spietato nell'evidenziare le anomalie**: il tuo valore è nell'audit
   critico, non nel rassicurare il management. Se i dati non tornano, dillo
   chiaramente e spiega perché.
""".strip()


# ─── Tool 1: RAG (LlamaIndex → ChromaDB) ────────────────────────────────────

def build_rag_tool() -> BaseTool:
    """Inizializza il tool RAG collegandosi a ChromaDB con LlamaIndex.

    Crea un VectorStoreIndex dalla collection "finsecure_docs" già popolata
    e lo espone come BaseTool.

    Returns:
        BaseTool LangChain che esegue query semantiche sui report finanziari.

    Raises:
        RuntimeError: Se la collection ChromaDB non esiste o il DB è vuoto.
    """
    import chromadb
    from langchain_core.tools import tool as lc_tool
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.query_engine import BaseQueryEngine
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore

    log.info("Inizializzazione RAG tool (ChromaDB: %s) ...", CHROMA_PERSIST_DIR)

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        log.info("Collection '%s': %d documenti indicizzati",
                 CHROMA_COLLECTION_NAME, chroma_collection.count())
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB non raggiungibile o collection '{CHROMA_COLLECTION_NAME}' assente. "
            f"Esegui prima 'rag_pipeline/ingest.py'. Errore: {exc}"
        ) from exc

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    llm = LlamaOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    Settings.embed_model = embed_model
    Settings.llm = llm

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine: BaseQueryEngine = index.as_query_engine(
        similarity_top_k=6,
        response_mode="tree_summarize",
    )

    @lc_tool
    def search_financial_reports(query: str) -> str:
        """Usa questo strumento per cercare informazioni, KPI e dichiarazioni \
nei report finanziari aziendali. Cerca sempre in modo estensivo per incrociare \
i dati. Adatto per: trovare valori di KPI (EBITDA, Total Debt, margini), \
verificare dichiarazioni di compliance, leggere sezioni specifiche dei report, \
confrontare dati tra documenti diversi."""
        try:
            response = query_engine.query(query)
            return str(response)
        except Exception as exc:
            log.error("Errore RAG query '%s': %s", query, exc)
            return f"Errore nella ricerca: {exc}"

    log.info("RAG tool inizializzato.")
    return search_financial_reports


# Wrapper Agent

class FinSecureAgent:
    """
    Avvia il processo MCP (risk_simulator.py) tramite stdio, carica i tool
    RAG e MCP, costruisce il grafo LangGraph con MemorySaver e lo espone
    tramite `ainvoke`.

    Utilizzo:
        async with FinSecureAgent() as agent:
            risposta = await agent.ainvoke(
                user_message="Verifica la coerenza del Total Debt",
                thread_id="sessione-audit-001",
            )
    """

    def __init__(self) -> None:
        self._mcp_client: Any = None
        self._graph: Any = None
        self._memory: Any = None

    async def __aenter__(self) -> "FinSecureAgent":
        """Inizializza MCP client, tool e grafo LangGraph."""
        _load_env()

        from langchain.agents.factory import create_agent
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.sessions import StdioConnection
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import MemorySaver

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        log.info("LLM: gpt-4o (temperature=0.1)")

        try:
            rag_tool = build_rag_tool()
            log.info("Tool RAG: OK")
        except RuntimeError as exc:
            log.warning("RAG tool non disponibile: %s", exc)
            log.warning("Agente avviato SENZA tool RAG. Esegui ingest.py prima.")
            rag_tool = None

        log.info("Caricamento tool MCP da: %s ...", MCP_SERVER_PATH)
        mcp_connection = StdioConnection(
            transport="stdio",
            command=sys.executable,
            args=["-u", MCP_SERVER_PATH],
        )
        self._mcp_client = MultiServerMCPClient(
            connections={"risk_simulator": mcp_connection}
        )
        _all_mcp: list[BaseTool] = await self._mcp_client.get_tools()
        # save_finding è escluso dall'agente: il salvataggio avviene in
        # background dopo che la risposta è stata mostrata all'utente.
        mcp_tools: list[BaseTool] = [t for t in _all_mcp if t.name != "save_finding"]
        log.info("Tool MCP caricati: %s", [t.name for t in mcp_tools])

        all_tools: list[BaseTool] = mcp_tools
        if rag_tool is not None:
            all_tools = [rag_tool] + mcp_tools
        log.info("Tool totali: %d  %s", len(all_tools), [t.name for t in all_tools])

        # Memoria conversazionale
        self._memory = MemorySaver()

        # Creazione agente
        self._graph = create_agent(
            model=llm,
            tools=all_tools,
            system_prompt=AUDIT_SYSTEM_PROMPT,
            checkpointer=self._memory,
            response_format=AuditResponse,
        )
        log.info("Agente LangGraph creato e pronto.")
        return self

    async def __aexit__(self, *_: Any) -> None:
        """
        Libera le risorse dell'agente.
        """
        self._mcp_client = None
        log.info("FinSecureAgent chiuso.")

    async def ainvoke(
        self,
        user_message: str,
        chat_history: list[BaseMessage] | None = None,
        thread_id: str = "default",
    ) -> dict[str, Any]:
        """Invia un messaggio all'agente e restituisce la risposta strutturata.

        Returns:
            Dict con chiavi:
            - "analysis": str — testo dell'analisi da mostrare all'utente
            - "findings": list[dict] — anomalie [{severity, description, source_docs}]
        """
        if self._graph is None:
            raise RuntimeError(
                "FinSecureAgent non inizializzato. Usa 'async with FinSecureAgent() as agent'."
            )

        messages: list[BaseMessage] = list(chat_history or [])
        messages.append(HumanMessage(content=user_message))
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        try:
            result = await self._graph.ainvoke(
                {"messages": messages},
                config=config,
            )
        except Exception as exc:
            log.error("Errore durante l'invocazione dell'agente: %s", exc)
            raise

        # Structured output
        structured = result.get("structured_response")
        if structured is not None:
            if hasattr(structured, "model_dump"):
                return structured.model_dump()
            if isinstance(structured, dict):
                return structured

        # Fallback: estrae testo dall'ultimo messaggio
        output_messages: list[Any] = result.get("messages", [])
        text = "Nessuna risposta generata dall'agente."
        if output_messages:
            last = output_messages[-1]
            if hasattr(last, "content"):
                text = str(last.content)
        return {"analysis": text, "findings": []}


def _load_env() -> None:
    """Carica .env e verifica le chiavi obbligatorie."""
    env_path = ROOT_DIR / ".env"
    load_dotenv(dotenv_path=env_path)

    missing = [k for k in ("OPENAI_API_KEY",) if not os.getenv(k)]
    if missing:
        log.error("Variabili d'ambiente mancanti: %s", ", ".join(missing))
        sys.exit(1)
