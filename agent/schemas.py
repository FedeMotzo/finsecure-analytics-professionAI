"""
agent/schemas.py — Schema Pydantic per l'output strutturato dell'agente.

Usato come `response_format` in `create_agent()`: il modello produce JSON
validato da Pydantic alla fine del reasoning, senza tool aggiuntivi.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Singola anomalia individuata durante l'audit."""

    severity: Literal["CRITICO", "ALTO", "MEDIO"] = Field(
        description="Livello di severità: CRITICO, ALTO, MEDIO"
    )
    description: str = Field(
        max_length=300,
        description="Descrizione concisa con valori numerici coinvolti"
    )
    source_docs: str = Field(
        description="Nome/i del PDF sorgente separati da virgola"
    )


class AuditResponse(BaseModel):
    """Risposta strutturata dell'agente di audit finanziario."""

    analysis: str = Field(
        description="Analisi completa e conversazionale da mostrare all'utente. "
        "Include: fatti accertati con valori numerici, RED FLAG con severità, "
        "raccomandazioni operative.",
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Lista di anomalie/red flag individuate. "
        "Vuota se non sono state trovate criticità.",
    )
