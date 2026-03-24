"""
tools/risk_simulator.py — Server MCP per la simulazione del rischio finanziario.

Espone strumenti di calcolo isolati che l'agente LangChain invoca via MCP.
Ogni tool è una funzione deterministica che riceve parametri
numerici e restituisce una stringa JSON strutturata.
"""

import json
import logging
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

mcp = FastMCP("FinSecure Risk Simulator")

DB_PATH = Path(__file__).parent.parent / "audit.db"


def _ensure_findings_table() -> None:
    """Crea la tabella findings se non esiste (idempotente)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Simulazione impatto sui tassi di interesse
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def simulate_rate_impact(
    current_operating_margin: float,
    rate_increase_bps: int,
) -> str:
    """Simula l'impatto di un aumento dei tassi di interesse sul margine operativo aziendale.

    Utilizza questo strumento ogni volta che l'utente chiede di valutare la
    sensibilità della redditività operativa a variazioni dei tassi BCE/Fed,
    oppure quando occorre eseguire uno stress test sul conto economico in
    scenari di rialzo dei tassi (es. "cosa succede al margine se i tassi
    salgono di 150 bps?").

    Modello di calcolo:
        Per ogni 100 basis points (bps) di aumento dei tassi, il margine
        operativo si riduce dello 0,15% sul valore corrente. La riduzione è
        proporzionale all'entità del rialzo:

            reduction_pct  = (rate_increase_bps / 100) * 0.0015
            new_margin     = current_operating_margin * (1 - reduction_pct)

        Questo modello approssima l'effetto combinato di:
        - Aumento del costo del debito a tasso variabile (passivo)
        - Compressione della domanda dei clienti in contesti ad alto costo del credito
        - Incremento degli oneri di rifinanziamento a breve termine

    Livello di rischio:
        "ALTO"  — se il nuovo margine operativo scende sotto € 10,0 milioni.
                  Indica che l'azienda si avvicina alla soglia di breakeven
                  operativo e potrebbe violare i covenant bancari.
        "BASSO" — se il nuovo margine rimane pari o superiore a € 10,0 milioni.
                  L'azienda mantiene un buffer operativo sufficiente.

    Args:
        current_operating_margin: Margine operativo corrente espresso in milioni
            di Euro (€ M). Deve essere un valore positivo. Esempio: 6.09 per
            un EBIT di € 6,09 milioni come riportato nel Report Q3 2025.
            Range atteso: 0.0 – 500.0.
        rate_increase_bps: Incremento dei tassi di interesse in basis points
            (1 bps = 0,01%). Deve essere un intero non negativo. Valori tipici:
            25 (rialzo ordinario), 50 (rialzo moderato), 100–200 (scenario
            avverso), 300+ (scenario estremo da stress test).
            Range atteso: 0 – 1000.

    Returns:
        Stringa JSON con i seguenti campi:
        - "current_margin_mln_eur": margine operativo iniziale (float, € M)
        - "rate_increase_bps": basis points di rialzo applicati (int)
        - "reduction_pct_applied": percentuale di riduzione applicata (float, %)
        - "new_margin_mln_eur": nuovo margine operativo post-shock (float, € M)
        - "risk_level": "ALTO" o "BASSO" (str)
        - "warning": messaggio descrittivo del livello di rischio (str)

    Esempio di output (input: margin=6.09, bps=200):
        {
            "current_margin_mln_eur": 6.09,
            "rate_increase_bps": 200,
            "reduction_pct_applied": 0.30,
            "new_margin_mln_eur": 5.77,
            "risk_level": "ALTO",
            "warning": "Margine operativo sotto soglia critica (< €10M): rischio di violazione covenant."
        }
    """
    if current_operating_margin < 0:
        return json.dumps({
            "error": "current_operating_margin deve essere un valore positivo.",
            "input_received": current_operating_margin,
        })
    if rate_increase_bps < 0:
        return json.dumps({
            "error": "rate_increase_bps deve essere un intero non negativo.",
            "input_received": rate_increase_bps,
        })

    reduction_pct: float = (rate_increase_bps / 100) * 0.0015
    new_margin: float = round(current_operating_margin * (1 - reduction_pct), 4)
    reduction_applied_pct: float = round(reduction_pct * 100, 4)

    risk_level: str
    warning: str
    if new_margin < 10.0:
        risk_level = "ALTO"
        warning = (
            f"Margine operativo sotto soglia critica (< €10M): rischio di violazione covenant "
            f"e compressione del buffer di liquidità operativa. "
            f"Considerare strategie di hedging sul debito a tasso variabile."
        )
    else:
        risk_level = "BASSO"
        warning = (
            f"Margine operativo entro limiti accettabili (≥ €10M). "
            f"Il rialzo di {rate_increase_bps} bps è assorbibile senza interventi strutturali immediati."
        )

    return json.dumps({
        "current_margin_mln_eur": current_operating_margin,
        "rate_increase_bps": rate_increase_bps,
        "reduction_pct_applied": reduction_applied_pct,
        "new_margin_mln_eur": new_margin,
        "risk_level": risk_level,
        "warning": warning,
    }, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Verifica covenant bancario Debt/EBITDA
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def check_debt_covenant(
    total_debt: float,
    ebitda: float,
) -> str:
    """Calcola il rapporto Debito/EBITDA e verifica il rispetto del covenant bancario.

    Utilizza questo strumento ogni volta che l'utente chiede di verificare la
    sostenibilità del debito aziendale, la compliance ai covenant dei contratti
    di finanziamento, oppure quando occorre confrontare due valori di Total Debt
    provenienti da documenti diversi (es. il Memo di Audit riporta €1,2M mentre
    l'Analisi di Rischio riporta €3,5M — quale usare per la verifica del covenant?).

    È lo strumento corretto anche in questi scenari:
    - "L'azienda rischia di violare il covenant sul debito?"
    - "Con un EBITDA di X e un debito di Y, siamo in regola?"
    - "Qual è il livello massimo di indebitamento tollerabile?"

    Covenant di riferimento:
        La soglia standard nei contratti di finanziamento sindacato di FinSecure
        Analytics è Net Debt / EBITDA ≤ 2,5x (come indicato nel Memo di Audit
        Interno Q3 2025, sezione 5.1). Il superamento di tale soglia costituisce
        un Evento di Default tecnico che può attivare clausole di accelerazione
        del debito.

    Modello di calcolo:
        ratio = total_debt / ebitda

        Soglie di valutazione:
        - ratio ≤ 2.5x  → COVENANT RISPETTATO  (verde)
        - ratio  2.5x–3.5x → ZONA DI ATTENZIONE (giallo): monitoraggio intensivo
        - ratio > 3.5x  → COVENANT VIOLATO (rosso): default tecnico imminente

    Args:
        total_debt: Debito finanziario totale (Total Debt) espresso in milioni
            di Euro (€ M). Include finanziamenti bancari, obbligazioni, linee
            revolving utilizzate e leasing IFRS 16. Deve essere un valore ≥ 0.
            ATTENZIONE: nei documenti di FinSecure Analytics esistono due valori
            contrastanti — €1,2M (Memo di Audit) e €3,5M (Analisi di Rischio).
            Specificare sempre quale fonte si sta utilizzando.
        ebitda: EBITDA (Earnings Before Interest, Taxes, Depreciation and
            Amortization) espresso in milioni di Euro (€ M). Deve essere > 0
            per evitare divisione per zero. Usare il valore annualizzato (LTM)
            per confrontabilità con i covenant che si basano su periodi rolling
            di 12 mesi. Esempio: se Q3 EBITDA trimestrale è €6,54M, l'EBITDA
            annualizzato LTM stimato è ~€26,16M.

    Returns:
        Stringa JSON con i seguenti campi:
        - "total_debt_mln_eur": debito totale in input (float, € M)
        - "ebitda_mln_eur": EBITDA in input (float, € M)
        - "debt_to_ebitda_ratio": rapporto calcolato arrotondato a 2 decimali (float)
        - "covenant_threshold": soglia di covenant applicata (float, default 2.5)
        - "status": "RISPETTATO", "ATTENZIONE" o "VIOLATO" (str)
        - "alert": messaggio descrittivo con raccomandazione operativa (str)
        - "max_sustainable_debt_mln_eur": debito massimo sostenibile entro covenant (float, € M)

    Esempio di output (input: debt=3.5, ebitda=6.54):
        {
            "total_debt_mln_eur": 3.5,
            "ebitda_mln_eur": 6.54,
            "debt_to_ebitda_ratio": 0.54,
            "covenant_threshold": 2.5,
            "status": "RISPETTATO",
            "alert": "Rapporto Debt/EBITDA (0.54x) ampiamente entro il covenant (2.5x).",
            "max_sustainable_debt_mln_eur": 16.35
        }

    Esempio di output con violazione (input: debt=3.5, ebitda=1.2):
        {
            "total_debt_mln_eur": 3.5,
            "ebitda_mln_eur": 1.2,
            "debt_to_ebitda_ratio": 2.92,
            "covenant_threshold": 2.5,
            "status": "VIOLATO",
            "alert": "⚠️ COVENANT VIOLATO: rapporto 2.92x supera la soglia di 2.5x. ..."
        }
    """
    COVENANT_THRESHOLD: float = 2.5
    ATTENTION_THRESHOLD: float = 3.5

    if ebitda <= 0:
        return json.dumps({
            "error": "ebitda deve essere > 0. Impossibile calcolare il rapporto Debt/EBITDA.",
            "input_received": {"total_debt": total_debt, "ebitda": ebitda},
        }, ensure_ascii=False)
    if total_debt < 0:
        return json.dumps({
            "error": "total_debt non può essere negativo.",
            "input_received": {"total_debt": total_debt, "ebitda": ebitda},
        }, ensure_ascii=False)

    ratio: float = round(total_debt / ebitda, 2)
    max_sustainable_debt: float = round(COVENANT_THRESHOLD * ebitda, 2)

    status: str
    alert: str

    if ratio <= COVENANT_THRESHOLD:
        status = "RISPETTATO"
        headroom: float = round(max_sustainable_debt - total_debt, 2)
        alert = (
            f"Rapporto Debt/EBITDA ({ratio}x) entro il covenant ({COVENANT_THRESHOLD}x). "
            f"Margine di headroom disponibile: €{headroom}M prima di raggiungere la soglia."
        )
    elif ratio <= ATTENTION_THRESHOLD:
        status = "ATTENZIONE"
        excess: float = round(total_debt - max_sustainable_debt, 2)
        alert = (
            f"⚠️ ZONA DI ATTENZIONE: rapporto {ratio}x supera il covenant ({COVENANT_THRESHOLD}x) "
            f"di €{excess}M. Monitoraggio trimestrale intensivo richiesto. "
            f"Comunicare proattivamente ai lead arrangers del finanziamento sindacato. "
            f"Valutare piano di deleveraging entro 6 mesi."
        )
    else:
        status = "VIOLATO"
        excess = round(total_debt - max_sustainable_debt, 2)
        alert = (
            f"🚨 COVENANT VIOLATO: rapporto {ratio}x supera la soglia critica ({ATTENTION_THRESHOLD}x). "
            f"Eccesso rispetto al covenant: €{excess}M. "
            f"Rischio di default tecnico: le banche possono richiedere il rimborso anticipato. "
            f"Azione immediata richiesta: contattare il CFO e i legali per notifica ai creditori. "
            f"Opzioni: iniezione di capitale, dismissione asset, waiver bancario."
        )

    result: dict = {
        "total_debt_mln_eur": total_debt,
        "ebitda_mln_eur": ebitda,
        "debt_to_ebitda_ratio": ratio,
        "covenant_threshold": COVENANT_THRESHOLD,
        "status": status,
        "alert": alert,
        "max_sustainable_debt_mln_eur": max_sustainable_debt,
    }

    return json.dumps(result, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Salvataggio criticità nel DB di audit
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def save_finding(
    severity: str,
    description: str,
    source_docs: str = "",
    session_id: str = "",
) -> str:
    """Salva una criticità o anomalia rilevata durante l'audit nel database SQLite.

    Chiama questo strumento OGNI VOLTA che individui una delle seguenti anomalie:
    - Errore di calcolo (es. totale dichiarato ≠ somma delle componenti)
    - Contraddizione tra documenti (es. stesso KPI con valori diversi in report diversi)
    - Omissione normativa (es. sezione GDPR incompleta o troncata)
    - Dato presentato in modo fuorviante (es. calo mascherato da linguaggio ottimistico)
    - Rischio non coperto (es. esposizione settoriale senza hedging)
    - Violazione di covenant o soglie regolamentari

    Il finding viene persistito nel DB SQLite e visualizzato in tempo reale
    nella dashboard della UI. Non chiamare questo tool per osservazioni neutre
    o fatti già noti — solo per anomalie che richiedono attenzione del management.

    Args:
        severity: Livello di severità dell'anomalia. Valori ammessi:
            - "CRITICO": impatto finanziario immediato o rischio legale grave
              (es. errore contabile, violazione GDPR, discrepanza debitoria)
            - "ALTO": rischio significativo che richiede azione entro 30 giorni
              (es. trend negativo mascherato, esposizione non coperta)
            - "MEDIO": anomalia da monitorare senza urgenza immediata
        description: Descrizione concisa e precisa dell'anomalia. Includere
            i valori numerici coinvolti (es. "EBITDA dichiarato €6.540K ≠
            ricalcolato €6.960K, delta €420K"). Max 300 caratteri.
        source_docs: Documento/i sorgente separati da virgola
            (es. "report_Q3.pdf, audit_compliance.pdf"). Lasciare vuoto
            se l'anomalia emerge dal confronto tra più documenti non specificati.
        session_id: Identificatore della sessione di audit corrente (thread_id
            LangGraph). Serve per raggruppare i findings per sessione nella
            dashboard. Se non disponibile, lasciare vuoto.

    Returns:
        Stringa JSON con conferma del salvataggio e id assegnato al finding.
        In caso di errore DB restituisce un JSON con campo "error".
    """
    allowed_severities = {"CRITICO", "ALTO", "MEDIO"}
    if severity.upper() not in allowed_severities:
        return json.dumps({
            "error": f"Severity non valida: '{severity}'. Valori ammessi: {allowed_severities}",
        }, ensure_ascii=False)

    if not description.strip():
        return json.dumps({"error": "description non può essere vuota."}, ensure_ascii=False)

    try:
        _ensure_findings_table()
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT INTO findings (session_id, severity, description, source_docs) "
                "VALUES (?, ?, ?, ?)",
                (session_id, severity.upper(), description.strip(), source_docs.strip()),
            )
            finding_id = cursor.lastrowid

        return json.dumps({
            "saved": True,
            "finding_id": finding_id,
            "severity": severity.upper(),
            "message": f"Finding #{finding_id} salvato nel DB di audit.",
        }, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({"error": f"Errore DB: {exc}"}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
