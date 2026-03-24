import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import gradio as gr
import nest_asyncio
from langchain_core.messages import AIMessage, HumanMessage


nest_asyncio.apply()

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agent.agent_core import FinSecureAgent

# Agente singleton inizializzato all'avvio dell'app

async def _init_agent() -> FinSecureAgent:
    agent = FinSecureAgent()
    await agent.__aenter__()
    return agent

_agent: FinSecureAgent = asyncio.get_event_loop().run_until_complete(_init_agent())


# DB
DB_PATH = ROOT_DIR / "audit.db"
STATUS_ICON = {"ok": "✅", "warning": "⚠️", "critical": "🔴"}
SEV_ICON    = {"CRITICO": "🔴", "ALTO": "🟡", "MEDIO": "🟠"}


def load_kpi_rows() -> list[list]:
    """Legge i KPI dalla tabella `kpis` di SQLite."""
    if not DB_PATH.exists():
        return [["DB non trovato — esegui ingest.py", "", "⚠️"]]
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT label, value, status FROM kpis ORDER BY id"
        ).fetchall()
    return [[r[0], r[1], STATUS_ICON.get(r[2], "❓")] for r in rows if r[1] != "N/D"] \
        or [["Nessun KPI nel DB", "", "⚠️"]]


def load_findings_rows() -> list[list]:
    """Legge le criticità dalla tabella `findings` di SQLite (ultime 20)."""
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT severity, description, source_docs, found_at "
            "FROM findings ORDER BY found_at DESC LIMIT 20"
        ).fetchall()
    return [
        [SEV_ICON.get(r[0], "⚪") + " " + r[0], r[1], r[2] or "—", r[3]]
        for r in rows
    ]


# Conversione chat history da Gradio a LangChain
def to_lc_history(history: list[dict]) -> list[Any]:
    """Converte la history di Gradio in messaggi LangChain"""
    msgs = []
    for turn in history[:-1]:
        if turn["role"] == "user":
            msgs.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            msgs.append(AIMessage(content=turn["content"]))
    return msgs


async def respond(
    message: str,
    history: list[dict],
    thread_id: str,
):
    """Async generator: mostra subito il messaggio utente, poi la risposta agente."""
    if not message.strip():
        yield "", history, thread_id
        return

    history = history + [{"role": "user", "content": message}]
    # Mostra subito l'indicatore di caricamento mentre l'agente elabora
    yield "", history + [{"role": "assistant", "content": "_⏳ Analisi in corso..._"}], thread_id

    try:
        result = await _agent.ainvoke(
            user_message=message,
            thread_id=thread_id,
        )
        answer = result["analysis"]
        findings = result.get("findings", [])
    except Exception as exc:
        answer = f"⚠️ Errore durante l'analisi: {exc}"
        findings = []

    # Mostra la risposta all'utente
    history = history + [{"role": "assistant", "content": answer}]
    yield "", history, thread_id

    # Salva i findings in background
    if findings:
        asyncio.create_task(_save_findings_to_db(findings, thread_id))


async def _save_findings_to_db(findings: list[dict], session_id: str) -> None:
    """Persiste i findings strutturati su SQLite (nessuna LLM call, solo INSERT)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS findings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL,
                    severity    TEXT    NOT NULL,
                    description TEXT    NOT NULL,
                    source_docs TEXT,
                    found_at    TEXT    DEFAULT (datetime('now','localtime'))
                )
            """)
            for f in findings:
                sev = str(f.get("severity", "")).upper()
                desc = str(f.get("description", "")).strip()[:300]
                src = str(f.get("source_docs", "")).strip()
                if sev in {"CRITICO", "ALTO", "MEDIO"} and desc:
                    conn.execute(
                        "INSERT INTO findings (session_id, severity, description, source_docs) "
                        "VALUES (?,?,?,?)",
                        (session_id, sev, desc, src),
                    )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Salvataggio findings fallito: %s", exc)


def new_session(_history: list) -> tuple[list, str]:
    import uuid
    return [], uuid.uuid4().hex


def fill_input(example: str) -> str:
    return example


def refresh_dashboard() -> tuple[list[list], list[list]]:
    return load_kpi_rows(), load_findings_rows()


# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
.header {
    background: linear-gradient(135deg, #0f2444 0%, #1a3a5c 100%);
    padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1rem;
    border-left: 4px solid #e8a020;
}
.header h1 { color: #fff; margin: 0; font-size: 1.35rem; }
.header p  { color: #a8c4db; margin: 0.25rem 0 0 0; font-size: 0.8rem; }
footer { display: none !important; }
"""

# ── Esempi di domande di audit ───────────────────────────────────────────────
EXAMPLES = [
    # Livello 1 — Trappole matematiche
    "Voglio un'analisi della nostra redditività nel terzo trimestre. Calcola l'EBITDA partendo dall'EBIT e dagli ammortamenti (D&A) riportati nel conto economico e dimmi se coincide con il KPI dell'EBITDA dichiarato in tabella.",
    # Livello 2 — Contraddizioni tra documenti
    "Qual è esattamente la nostra esposizione debitoria totale (Total Debt)? Cerca questo dato in tutti i report disponibili e dimmi se c'è coerenza tra le diverse divisioni aziendali.",
    # Livello 3 — Omissioni normative
    "Il Board vuole rassicurazioni sulla compliance GDPR. Siamo completamente a norma in tutte le nostre filiali europee? Leggi attentamente lo stato della conformità nel memo di audit.",
    # Livello 4 — Tool MCP: covenant
    "Prendi il valore del Total Debt più alto che riesci a trovare nei nostri report e l'EBITDA corretto che hai ricalcolato prima. Usa i tuoi strumenti di simulazione per verificare se rischiamo di violare i covenant bancari.",
    # Livello 4 — Tool MCP: tassi
    "Ho letto che abbiamo una forte esposizione nel settore Real Estate senza hedging. Considerando il nostro Margine Operativo attuale (EBIT in milioni), usa il simulatore per calcolare quale sarebbe il nuovo margine e il livello di rischio se la BCE dovesse alzare i tassi di 150 basis points.",
    # Livello 5 — Memoria conversazionale
    "E se invece lo shock sui tassi fosse solo di 50 basis points?",
]

#Layout
import uuid

with gr.Blocks(title="FinSecure Analytics — AI Audit Terminal") as demo:

    thread_state = gr.State(value=uuid.uuid4().hex)

    gr.HTML("""
        <div class='header'>
            <h1>📊 FinSecure Analytics — AI Audit Terminal</h1>
            <p>Chief Audit Executive AI &nbsp;·&nbsp; GPT-4o + LlamaIndex RAG + MCP Risk Simulator</p>
        </div>
    """)

    with gr.Row(equal_height=True):

        # Colonna chat
        with gr.Column(scale=65):

            chatbot = gr.Chatbot(
                label="Sessione di Audit",
                height=480,
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=finsecure",
                ),
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Chiedi all'agente di verificare un dato o simulare uno scenario...",
                    lines=1,
                    scale=9,
                    container=False,
                    submit_btn=False,
                )
                send_btn = gr.Button("Invia ➤", variant="primary", scale=1)

            with gr.Accordion("💡 Domande suggerite", open=False):
                for ex in EXAMPLES:
                    gr.Button(ex, size="sm").click(
                        fn=fill_input,
                        inputs=gr.State(ex),
                        outputs=msg_box,
                    )

            clear_btn = gr.Button("🗑️ Nuova sessione di audit", variant="secondary")

        # Colonna dashboard
        with gr.Column(scale=35):

            gr.Markdown("### 📡 Live Risk Monitor")

            with gr.Row():
                gr.Textbox(value="● ONLINE", label="Sistema",   interactive=False)
                gr.Textbox(value="● ACTIVE", label="MCP",       interactive=False)
                gr.Textbox(value="3",        label="Documenti", interactive=False)

            refresh_btn = gr.Button("🔄 Aggiorna Dashboard", size="sm")

            gr.Markdown("---")
            gr.Markdown("**📊 KPI Q3 2025**")
            gr.Markdown("Estratti automaticamente durante l'ingestion · fonte: audit.db")
            kpi_table = gr.Dataframe(
                value=load_kpi_rows(),
                headers=["KPI", "Valore", "Stato"],
                interactive=False,
            )

            gr.Markdown("---")
            gr.Markdown("**🚩 Criticità individuate**")
            gr.Markdown("Salvate dall'agente in tempo reale · premi Aggiorna per ricaricare")
            findings_table = gr.Dataframe(
                value=load_findings_rows(),
                headers=["Severità", "Descrizione", "Fonte", "Rilevata il"],
                interactive=False,
            )

    submit_inputs  = [msg_box, chatbot, thread_state]
    submit_outputs = [msg_box, chatbot, thread_state]

    msg_box.submit(fn=respond, inputs=submit_inputs, outputs=submit_outputs)
    send_btn.click(fn=respond, inputs=submit_inputs, outputs=submit_outputs)

    clear_btn.click(
        fn=new_session,
        inputs=[chatbot],
        outputs=[chatbot, thread_state],
    )

    refresh_btn.click(
        fn=refresh_dashboard,
        outputs=[kpi_table, findings_table],
    )

demo.launch(css=CSS, share=True, show_error=True, quiet=False)
