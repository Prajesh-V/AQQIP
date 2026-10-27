"""
AQQIP SEC EDGAR Company Facts Extractor

Extracts financial statement line items from SEC EDGAR XBRL company facts
to compute Beneish M-Score and Altman Z''-Score without lookahead bias.
Persists fundamental ratios to MySQL with actual publication (filing) dates.
"""
from __future__ import annotations

import datetime
import json
import time
from typing import Dict, List, Optional, Any
import requests

from config.settings import get_settings
from src.database.mysql_client import get_mysql_connection
from src.quant.altman_score import compute_altman_z_double_prime_score
from src.quant.beneish_score import compute_beneish_m_score
from src.utils.logging import get_logger
from src.utils.retry import with_http_retry

logger = get_logger("edgar_company_facts")


# Mapping of standard US-GAAP concepts to normalized financial fields
CONCEPT_MAP = {
    # Assets & Liabilities
    "Assets": "total_assets",
    "AssetsCurrent": "current_assets",
    "Liabilities": "total_liabilities",
    "LiabilitiesCurrent": "current_liabilities",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings",
    "StockholdersEquity": "stockholders_equity",
    "AccountsReceivableNetCurrent": "receivables",
    "PropertyPlantAndEquipmentNet": "ppe",
    "LongTermDebtNoncurrent": "long_term_debt",
    
    # Revenue / Sales
    "RevenueFromContractWithCustomerExcludingAssessedTax": "sales",
    "Revenues": "sales",
    "SalesRevenueNet": "sales",
    
    # Cost & Expenses
    "CostOfGoodsAndServicesSold": "cogs",
    "CostOfRevenue": "cogs",
    "OperatingIncomeLoss": "ebit",
    "SellingGeneralAndAdministrativeExpense": "sg_a",
    "GeneralAndAdministrativeExpense": "sg_a",
    "DepreciationDepletionAndAmortization": "depreciation",
    "DepreciationAndAmortization": "depreciation",
    "Depreciation": "depreciation",
    
    # Income & Cash Flow
    "NetIncomeLoss": "net_income",
    "NetCashProvidedByUsedInOperatingActivities": "cash_flow_op",
}


class EdgarCompanyFactsExtractor:
    """Extracts statement items and computes fundamental ratios from SEC XBRL."""

    def __init__(self):
        self.settings = get_settings()
        self.headers = {"User-Agent": self.settings.edgar.user_agent}

    @with_http_retry
    def _fetch_facts(self, cik: str) -> dict:
        """Fetch SEC company facts JSON with rate-limiting and retry."""
        cik_padded = str(int(cik)).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
        logger.info("Fetching SEC company facts", cik=cik_padded, url=url)
        time.sleep(0.2)  # Respect SEC rate limits (<= 10 requests/sec)
        resp = requests.get(url, headers=self.headers, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def extract_and_store(self, ticker: str, cik: str) -> int:
        """
        Fetch company facts for a ticker/cik, compute ratios, and upsert to MySQL.
        Returns the number of fundamental ratio records stored.
        """
        try:
            data = self._fetch_facts(cik)
        except Exception as e:
            logger.error("Failed to fetch facts from SEC", ticker=ticker, cik=cik, error=str(e))
            return 0

        us_gaap = data.get("facts", {}).get("us-gaap", {})
        if not us_gaap:
            logger.warning("No us-gaap facts found", ticker=ticker)
            return 0

        # Group facts by filing period key: (form, fy, fp, filed, accn)
        # We focus on 10-K and 10-Q filings
        periods: Dict[tuple, Dict[str, float]] = {}

        for concept_name, target_field in CONCEPT_MAP.items():
            if concept_name not in us_gaap:
                continue

            units = us_gaap[concept_name].get("units", {})
            for unit_key, entries in units.items():
                for entry in entries:
                    form = entry.get("form")
                    if form not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                        continue

                    filed_str = entry.get("filed")
                    fy = entry.get("fy")
                    fp = entry.get("fp")
                    accn = entry.get("accn")
                    val = entry.get("val")

                    if not (filed_str and fy and fp and accn and val is not None):
                        continue

                    # Filter to 10+ years of historical financial filings (from 2013 onwards)
                    try:
                        fy_int = int(fy)
                        if fy_int < 2013:
                            continue
                    except (ValueError, TypeError):
                        pass

                    key = (form, fy, fp, filed_str, accn)
                    if key not in periods:
                        periods[key] = {}

                    # Only set if not already set or prioritize more specific entries
                    if target_field not in periods[key]:
                        periods[key][target_field] = float(val)

        if not periods:
            logger.warning("No relevant 10-K/10-Q periods found", ticker=ticker)
            return 0

        # Sort periods chronologically by filing date (publication date)
        sorted_keys = sorted(periods.keys(), key=lambda k: k[3])  # k[3] is filed_str

        records_to_insert = []
        prior_annual: Optional[Dict[str, float]] = None

        for form, fy, fp, filed_str, accn in sorted_keys:
            raw = periods[(form, fy, fp, filed_str, accn)]
            fiscal_period_str = f"{fy}-{fp}"
            filing_date = datetime.date.fromisoformat(filed_str)

            # 1. Compute Altman Z''-Score
            # Requires: current_assets, current_liabilities, retained_earnings, ebit, book_value_equity, total_liabilities, total_assets
            altman_val: Optional[float] = None
            total_assets = raw.get("total_assets")
            current_assets = raw.get("current_assets")
            current_liabilities = raw.get("current_liabilities")
            retained_earnings = raw.get("retained_earnings")
            ebit = raw.get("ebit")
            equity = raw.get("stockholders_equity")
            total_liabilities = raw.get("total_liabilities")

            if total_liabilities is None and total_assets is not None and equity is not None:
                # Standard accounting identity: Liabilities = Assets - Equity
                total_liabilities = total_assets - equity

            if None not in (current_assets, current_liabilities, retained_earnings, ebit, equity, total_liabilities, total_assets) and total_assets > 0:
                altman_res = compute_altman_z_double_prime_score(
                    current_assets=current_assets,
                    current_liabilities=current_liabilities,
                    retained_earnings=retained_earnings,
                    ebit=ebit,
                    book_value_equity=equity,
                    total_liabilities=total_liabilities,
                    total_assets=total_assets
                )
                altman_val = altman_res.get("Z_DOUBLE_PRIME")

            # 2. Compute Beneish M-Score (primarily on annual 10-K reports with prior period)
            beneish_val: Optional[float] = None
            if form == "10-K" and prior_annual is not None:
                curr_sales = raw.get("sales")
                prior_sales = prior_annual.get("sales")
                curr_cogs = raw.get("cogs", 0.0)
                prior_cogs = prior_annual.get("cogs", 0.0)
                curr_rec = raw.get("receivables", 0.0)
                prior_rec = prior_annual.get("receivables", 0.0)
                curr_ca = raw.get("current_assets", 0.0)
                prior_ca = prior_annual.get("current_assets", 0.0)
                curr_ppe = raw.get("ppe", 0.0)
                prior_ppe = prior_annual.get("ppe", 0.0)
                curr_dep = raw.get("depreciation", 0.0)
                prior_dep = prior_annual.get("depreciation", 0.0)
                curr_ta = raw.get("total_assets")
                prior_ta = prior_annual.get("total_assets")
                curr_sga = raw.get("sg_a", 0.0)
                prior_sga = prior_annual.get("sg_a", 0.0)
                curr_ni = raw.get("net_income", 0.0)
                curr_cfo = raw.get("cash_flow_op", 0.0)
                curr_cl = raw.get("current_liabilities", 0.0)
                prior_cl = prior_annual.get("current_liabilities", 0.0)
                curr_ltd = raw.get("long_term_debt", 0.0)
                prior_ltd = prior_annual.get("long_term_debt", 0.0)

                if curr_sales and prior_sales and curr_ta and prior_ta:
                    b_res = compute_beneish_m_score(
                        current_sales=curr_sales, prior_sales=prior_sales,
                        current_cogs=curr_cogs, prior_cogs=prior_cogs,
                        current_receivables=curr_rec, prior_receivables=prior_rec,
                        current_current_assets=curr_ca, prior_current_assets=prior_ca,
                        current_ppe=curr_ppe, prior_ppe=prior_ppe,
                        current_depreciation=curr_dep, prior_depreciation=prior_dep,
                        current_total_assets=curr_ta, prior_total_assets=prior_ta,
                        current_sg_a=curr_sga, prior_sg_a=prior_sga,
                        current_net_income=curr_ni, current_cash_flow_op=curr_cfo,
                        current_current_liabilities=curr_cl, prior_current_liabilities=prior_cl,
                        current_long_term_debt=curr_ltd, prior_long_term_debt=prior_ltd
                    )
                    beneish_val = b_res.get("M_SCORE")

            if form == "10-K":
                prior_annual = raw

            # Calculate feature completeness ratio
            total_expected_keys = len(set(CONCEPT_MAP.values()))
            found_keys = len(raw)
            completeness = round(found_keys / total_expected_keys, 4)

            records_to_insert.append({
                "ticker": ticker.upper(),
                "cik": str(cik).zfill(10),
                "fiscal_period": fiscal_period_str,
                "filing_date": filing_date,
                "beneish_m_score": beneish_val,
                "altman_z_score": altman_val,
                "raw_features_json": json.dumps(raw),
                "source_filing_accession": accn,
                "feature_completeness": completeness,
            })

        if not records_to_insert:
            return 0

        # Upsert into MySQL fundamental_ratios
        query = """
            INSERT INTO fundamental_ratios (
                ticker, cik, fiscal_period, filing_date, beneish_m_score,
                altman_z_score, raw_features_json, source_filing_accession, feature_completeness
            ) VALUES (
                %(ticker)s, %(cik)s, %(fiscal_period)s, %(filing_date)s, %(beneish_m_score)s,
                %(altman_z_score)s, %(raw_features_json)s, %(source_filing_accession)s, %(feature_completeness)s
            ) ON DUPLICATE KEY UPDATE
                filing_date = VALUES(filing_date),
                beneish_m_score = VALUES(beneish_m_score),
                altman_z_score = VALUES(altman_z_score),
                raw_features_json = VALUES(raw_features_json),
                source_filing_accession = VALUES(source_filing_accession),
                feature_completeness = VALUES(feature_completeness),
                computed_at = CURRENT_TIMESTAMP
        """

        with get_mysql_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, records_to_insert)
            conn.commit()

        logger.info(
            "Persisted fundamental ratios",
            ticker=ticker,
            count=len(records_to_insert),
            altman_count=sum(1 for r in records_to_insert if r["altman_z_score"] is not None),
            beneish_count=sum(1 for r in records_to_insert if r["beneish_m_score"] is not None)
        )
        return len(records_to_insert)
