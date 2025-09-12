from __future__ import annotations

from typing import Any, List, Dict
from pydantic import BaseModel, Field


class ImagingVerdict(BaseModel):
    is_imaging: bool
    modalities: List[str] = Field(default_factory=list)
    confidence: float
    reasons: List[str] = Field(default_factory=list)
    counter_signals: List[str] = Field(default_factory=list)


class TitleResponse(BaseModel):
    title: str
    confidence: float
    reasons: List[str] = Field(default_factory=list)


class PageClassResponse(BaseModel):
    labels: List[str]
    modalities: List[str] = Field(default_factory=list)
    score: float
    evidence: List[str] = Field(default_factory=list)


class ExtractionCandidate(BaseModel):
    field: str
    page: int
    raw_span: str
    value: Any
    units: str = ""
    evidence: str
    confidence: float = 0.0
    notes: str | None = None


class ExtractionCandidates(BaseModel):
    candidates: List[ExtractionCandidate] = Field(default_factory=list)


class AdjudicatedFieldEntry(BaseModel):
    # Value types vary; we coerce downstream
    value: Any
    units: str | None = None
    page: int | None = None
    evidence: str | None = None
    confidence: float | None = None
    reason: str | None = None


class AdjudicationResponse(BaseModel):
    fields: Dict[str, AdjudicatedFieldEntry] = Field(default_factory=dict)

