from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class SecurityIdentity(BaseModel):
    ticker: str
    company_name: str
    cik: str
    exchange: Optional[str] = None
    security_type: str = "common_stock"

class MarketContext(BaseModel):
    current_price: float
    recent_return: float
    recent_volatility: float
    average_volume: float
    liquidity_proxy: float
    as_of_date: date

class QuantitativeSignalBlock(BaseModel):
    quant_pressure_score: float
    beneish_m_score: Optional[float] = None
    altman_z_score: Optional[float] = None
    model_probability: Optional[float] = None
    model_run_id: str
    model_variant: str = "M"
    input_feature_hash: Optional[str] = None
    features_snapshot: Optional[Dict[str, Optional[float]]] = None

class QualitativeSignalBlock(BaseModel):
    semantic_shift_score: float = 0.0
    cosine_similarity: float = 1.0
    change_magnitude: float = 0.0
    directional_risk_score: float = 0.0
    added_count: int = 0
    deleted_count: int = 0
    shifted_count: int = 0
    filing_year_from: Optional[int] = None
    filing_year_to: Optional[int] = None
    source_accession: Optional[str] = None
    prior_accession: Optional[str] = None
    filing_availability_date: Optional[date] = None
    evidence_snippets: List[str] = Field(default_factory=list)
    extraction_status: str = "SUCCESS"

class EvidenceQuality(BaseModel):
    quant_available: bool
    qualitative_available: bool
    filing_pair_available: bool
    feature_completeness: float
    data_freshness_minutes: Optional[float] = None
    missing_reasons: List[str] = Field(default_factory=list)

class FusionBlock(BaseModel):
    evidence_concordance: float
    combined_forensic_pressure: float
    fusion_version: str = "v1.0.0"

class ProvenanceBlock(BaseModel):
    as_of_date: Optional[date] = None
    prediction_id: Optional[int] = None
    quant_model_run_id: Optional[str] = None
    qualitative_diff_id: Optional[str] = None
    random_seed: int = 42
    scenario_id: Optional[str] = None

class SignalBundle(BaseModel):
    security: SecurityIdentity
    market_context: MarketContext
    quantitative: QuantitativeSignalBlock
    qualitative: QualitativeSignalBlock
    evidence_quality: EvidenceQuality
    fusion: FusionBlock
    provenance: ProvenanceBlock

class MARLResult(BaseModel):
    simulation_id: str
    ticker: str
    scenario_seed: int
    episode_days: int
    initial_price: float
    final_price: float
    max_drawdown_pct: float
    price_volatility: float
    short_seller_pnl_pct: float
    retail_herd_pnl_pct: float
    market_maker_pnl_pct: float
    market_maker_inventory_peak: float
    terminated_reason: str
    action_trace_uri_or_reference: Optional[str] = None
    scenario_params: dict
    signal_bundle_hash: str
    model_version: str
    reward_config_version: str = "v1.0.0"
    prediction_id: Optional[int] = None
    runtime_ms: Optional[float] = None
    run_at: datetime = Field(default_factory=datetime.utcnow)
