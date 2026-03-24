"""
eval/benchmark.py — Benchmark delle 6 anomalie sui documenti FinSecure Analytics.

Esegue una domanda per ciascuna trappola intenzionale, raccoglie la risposta
strutturata dell'agente e salva i risultati in eval/benchmark_results.json.
"""

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")
sys.path.insert(0, str(ROOT_DIR))

from agent.agent_core import FinSecureAgent

QUESTIONS = [
    {
        "id": "Q1",
        "trap": "T1 — Errore matematico EBITDA",
        "question": (
            "Voglio verificare la correttezza dell'EBITDA dichiarato nel report Q3. "
            "Partendo dal conto economico, calcola autonomamente l'EBITDA sommando "
            "EBIT e ammortamenti (D&A) e dimmi se il risultato coincide con il KPI "
            "riportato in tabella."
        ),
    },
    {
        "id": "Q2",
        "trap": "T2 — FCF presentato ottimisticamente",
        "question": (
            "Il CFO afferma che il calo del Free Cash Flow nel Q3 è una "
            "'fisiologica normalizzazione stagionale'. "
            "Analizza i dati effettivi del flusso di cassa e dimmi se "
            "questa valutazione è corretta o fuorviante."
        ),
    },
    {
        "id": "Q3",
        "trap": "T3 — Sezione GDPR troncata",
        "question": (
            "Il Board mi chiede conferma scritta che siamo pienamente conformi "
            "al GDPR in tutte le nostre sedi europee, incluse le filiali di "
            "Francia e Irlanda. Puoi confermarlo leggendo il memo di audit?"
        ),
    },
    {
        "id": "Q4",
        "trap": "T4+T5 — Contraddizione Total Debt tra documenti",
        "question": (
            "Ho bisogno del dato preciso sul debito finanziario totale dell'azienda "
            "per una presentazione agli investitori. "
            "Cerca questo numero in tutti i documenti disponibili e dimmi "
            "se esiste un valore univoco e affidabile."
        ),
    },
    {
        "id": "Q5",
        "trap": "T6 — Real Estate senza hedging",
        "question": (
            "Il Risk Committee vuole sapere se l'esposizione al settore Real Estate "
            "è adeguatamente coperta da strumenti di hedging. "
            "Cosa emerge dall'analisi di rischio di mercato?"
        ),
    },
    {
        "id": "Q6",
        "trap": "T1+T4/T5 — Stress test covenant con dati corretti",
        "question": (
            "Usando il valore di debito più alto (worst case) che riesci a trovare "
            "nei nostri report e l'EBITDA correttamente ricalcolato da EBIT e D&A, "
            "usa il simulatore per verificare se siamo a rischio di violare "
            "il covenant bancario Debt/EBITDA 2.5x."
        ),
    },
]

async def run_benchmark() -> None:
    output_path = ROOT_DIR / "eval" / "benchmark_results.json"
    results = []

    print("=" * 65)
    print("  FinSecure Analytics — Benchmark Anomalie")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    async with FinSecureAgent() as agent:
        for q in QUESTIONS:
            thread_id = uuid.uuid4().hex
            print(f"\n[{q['id']}] {q['trap']}")
            print(f"  thread_id: {thread_id}")
            print(f"  Domanda  : {q['question'][:80]}...")

            try:
                result = await agent.ainvoke(
                    user_message=q["question"],
                    thread_id=thread_id,
                )
                analysis = result.get("analysis", "")
                findings = result.get("findings", [])

                print(f"  Findings : {len(findings)} anomalia/e rilevata/e")
                for f in findings:
                    sev = f.get("severity", "?")
                    desc = f.get("description", "")[:70]
                    print(f"    [{sev}] {desc}")

                results.append({
                    "id": q["id"],
                    "trap": q["trap"],
                    "thread_id": thread_id,
                    "question": q["question"],
                    "analysis": analysis,
                    "findings": findings,
                    "findings_count": len(findings),
                    "timestamp": datetime.now().isoformat(),
                })

            except Exception as exc:
                print(f"  ERRORE: {exc}")
                results.append({
                    "id": q["id"],
                    "trap": q["trap"],
                    "thread_id": thread_id,
                    "question": q["question"],
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(),
                })

    # Salva risultati
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Riepilogo
    print("\n" + "=" * 65)
    print("  RIEPILOGO")
    print("=" * 65)
    total_findings = sum(r.get("findings_count", 0) for r in results)
    errors = sum(1 for r in results if "error" in r)
    print(f"  Domande eseguite : {len(results)}/{len(QUESTIONS)}")
    print(f"  Anomalie rilevate: {total_findings}")
    print(f"  Errori           : {errors}")
    print(f"\n  Risultati salvati in: {output_path}")
    print("  Usa i thread_id per trovare le trace su LangSmith.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
