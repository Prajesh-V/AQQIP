"""
AQQIP Streamlit Dashboard — Phase 1 Stub

This is a minimal placeholder that verifies Streamlit starts correctly.
All business views will be implemented in Phase 9.

This file exists solely to satisfy the Phase 1 Docker infrastructure requirement
that the streamlit_dashboard container starts and passes its healthcheck.
"""

import streamlit as st

st.set_page_config(
    page_title="AQQIP — Research Platform",
    page_icon="🔬",
    layout="wide",
)

st.title("AQQIP — Autonomous Quantitative & Qualitative Investment Platform")

st.info(
    "⚙️ **Phase 1 — Infrastructure**\n\n"
    "The dashboard is a placeholder. "
    "Business views will be implemented in Phase 9 after all analysis engines are built."
)

st.caption(
    "⚠️ AQQIP is a financial research and simulation tool. "
    "Outputs are model-generated signals and simulated scenarios, "
    "not investment advice or fraud verdicts."
)
