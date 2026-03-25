import asyncio
import io
import logging
import os
import re
import sqlite3
import sys
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
import nest_asyncio

nest_asyncio.apply()

# ── Path del progetto ────────────────────────────────────────────────────────
try:
    ROOT_DIR = Path(__file__).parent.parent
except NameError:
    ROOT_DIR = Path("/content/finsecure-analytics-professionAI")

sys.path.insert(0, str(ROOT_DIR))

# ── Variabili d'ambiente ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ROOT_DIR / ".env")
except Exception:
    pass

# ── Colab fix: ipykernel OutStream.fileno() ──────────────────────────────────
try:
    from ipykernel.iostream import OutStream as _OutStream

    if not getattr(_OutStream, "_fileno_patched", False):
        _orig_fileno = _OutStream.fileno
        _devnull_fd = os.open(os.devnull, os.O_RDWR)

        def _safe_fileno(self):
            try:
                return _orig_fileno(self)
            except io.UnsupportedOperation:
                return _devnull_fd

        _OutStream.fileno = _safe_fileno
        _OutStream._fileno_patched = True
except (ImportError, Exception):
    pass

# ── Inizializzazione agente ──────────────────────────────────────────────────
from agent.agent_core import FinSecureAgent

log = logging.getLogger(__name__)


async def _init_agent() -> FinSecureAgent:
    agent = FinSecureAgent()
    await agent.__aenter__()
    return agent


print("Inizializzazione agente in corso...")
_agent = asyncio.get_event_loop().run_until_complete(_init_agent())
print("Agente pronto.")

# ── DB helpers ───────────────────────────────────────────────────────────────
DB_PATH = ROOT_DIR / "audit.db"
STATUS_ICON = {"ok": "\u2705", "warning": "\u26a0\ufe0f", "critical": "\U0001f534"}
SEV_ICON = {"CRITICO": "\U0001f534", "ALTO": "\U0001f7e1", "MEDIO": "\U0001f7e0"}


def load_kpi_rows() -> list[list]:
    if not DB_PATH.exists():
        return [["DB non trovato \u2014 esegui ingest.py", "", "\u26a0\ufe0f"]]
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT label, value, status FROM kpis ORDER BY id"
        ).fetchall()
    return [
        [r[0], r[1], STATUS_ICON.get(r[2], "\u2753")] for r in rows if r[1] != "N/D"
    ] or [["Nessun KPI nel DB", "", "\u26a0\ufe0f"]]


def load_findings_rows() -> list[list]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT severity, description, source_docs, found_at "
            "FROM findings ORDER BY found_at DESC LIMIT 20"
        ).fetchall()
    return [
        [SEV_ICON.get(r[0], "\u26aa") + " " + r[0], r[1], r[2] or "\u2014", r[3]]
        for r in rows
    ]


def _parse_numeric(value_str: str) -> float | None:
    """Estrae un valore numerico da stringhe KPI come '€142.5M' o '6.540 K€'."""
    if not value_str or value_str == "N/D":
        return None
    clean = value_str.replace("\u20ac", "").replace("$", "").replace(" ", "")
    # Gestione separatore migliaia italiano (.) vs decimale
    # Se c'è un solo punto e cifre dopo, trattalo come decimale
    # Se c'è una virgola seguita da cifre, quello è il decimale
    clean = clean.replace(",", ".")
    multiplier = 1.0
    upper = clean.upper()
    if upper.endswith("M"):
        multiplier = 1.0
        clean = clean[:-1]
    elif upper.endswith("K"):
        multiplier = 0.001
        clean = clean[:-1]
    # Rimuove tutto tranne cifre e punto
    clean = re.sub(r"[^\d.]", "", clean)
    if not clean:
        return None
    try:
        return float(clean) * multiplier
    except ValueError:
        return None


def build_kpi_chart():
    """Grafico a barre orizzontali dei KPI (valori in milioni)."""
    fig, ax = plt.subplots(figsize=(5, 3))
    if not DB_PATH.exists():
        ax.text(0.5, 0.5, "DB non trovato", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT label, value, status FROM kpis ORDER BY id").fetchall()

    labels, values, colors = [], [], []
    color_map = {"ok": "#22c55e", "warning": "#eab308", "critical": "#ef4444"}

    for label, value_str, status in rows:
        num = _parse_numeric(value_str)
        if num is not None:
            labels.append(label)
            values.append(num)
            colors.append(color_map.get(status, "#6b7280"))

    if not labels:
        ax.text(0.5, 0.5, "Nessun KPI numerico", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Valore (M\u20ac)")
    ax.set_title("KPI Finanziari Q3 2025", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def build_findings_chart():
    """Grafico a barre delle criticita per livello di severita."""
    fig, ax = plt.subplots(figsize=(5, 3))
    if not DB_PATH.exists():
        ax.text(0.5, 0.5, "DB non trovato", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT severity, COUNT(*) FROM findings GROUP BY severity"
        ).fetchall()

    if not rows:
        ax.text(0.5, 0.5, "Nessuna criticita rilevata", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    sev_order = {"CRITICO": 0, "ALTO": 1, "MEDIO": 2}
    rows = sorted(rows, key=lambda r: sev_order.get(r[0], 99))
    severities = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    sev_colors = {"CRITICO": "#ef4444", "ALTO": "#eab308", "MEDIO": "#f97316"}
    colors = [sev_colors.get(s, "#6b7280") for s in severities]

    bars = ax.bar(severities, counts, color=colors)
    ax.set_ylabel("Conteggio")
    ax.set_title("Criticita per Severita", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(c), ha="center", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig


async def _save_findings_to_db(findings: list[dict], session_id: str) -> None:
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
        log.warning("Salvataggio findings fallito: %s", exc)


# ── Handler chat ─────────────────────────────────────────────────────────────
async def respond(message, history, thread_id):
    if not message.strip():
        yield "", history, thread_id
        return

    history = history + [{"role": "user", "content": message}]
    history = history + [{"role": "assistant", "content": "Analisi in corso..."}]
    yield "", history, thread_id

    # Rimuove il placeholder prima di inserire la risposta reale
    history = history[:-1]

    try:
        result = await _agent.ainvoke(user_message=message, thread_id=thread_id)
        answer = result["analysis"]
        findings = result.get("findings", [])
    except Exception as exc:
        answer = f"Errore: {exc}"
        findings = []

    history = history + [{"role": "assistant", "content": answer}]
    yield "", history, thread_id

    if findings:
        asyncio.create_task(_save_findings_to_db(findings, thread_id))


def new_session():
    return [], uuid.uuid4().hex


def refresh_dashboard():
    plt.close("all")
    return load_kpi_rows(), build_kpi_chart(), load_findings_rows(), build_findings_chart()


# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="FinSecure Analytics") as demo:
    thread_state = gr.State(value=uuid.uuid4().hex)

    gr.Markdown("## FinSecure Analytics \u2014 AI Audit Terminal")

    with gr.Row(equal_height=True):

        # ── Colonna sinistra: chat ───────────────────────────────────────
        with gr.Column(scale=65):
            chatbot = gr.Chatbot(type="messages", height=500)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Scrivi una domanda di audit...",
                    scale=9,
                    container=False,
                )
                send = gr.Button("Invia", variant="primary", scale=1)

            clear = gr.Button("Nuova sessione")

        # ── Colonna destra: dashboard ────────────────────────────────────
        with gr.Column(scale=35):
            gr.Markdown("### KPI Q3 2025")
            kpi_chart = gr.Plot(value=build_kpi_chart())
            kpi_table = gr.Dataframe(
                value=load_kpi_rows(),
                headers=["KPI", "Valore", "Stato"],
                interactive=False,
            )

            gr.Markdown("### Criticita individuate")
            findings_chart = gr.Plot(value=build_findings_chart())
            findings_table = gr.Dataframe(
                value=load_findings_rows(),
                headers=["Severita", "Descrizione", "Fonte", "Rilevata il"],
                interactive=False,
            )

            refresh_btn = gr.Button("Aggiorna Dashboard")

    # ── Eventi ───────────────────────────────────────────────────────────
    msg.submit(respond, [msg, chatbot, thread_state], [msg, chatbot, thread_state])
    send.click(respond, [msg, chatbot, thread_state], [msg, chatbot, thread_state])
    clear.click(new_session, outputs=[chatbot, thread_state])
    refresh_btn.click(refresh_dashboard, outputs=[kpi_table, kpi_chart, findings_table, findings_chart])

demo.launch(share=True)
