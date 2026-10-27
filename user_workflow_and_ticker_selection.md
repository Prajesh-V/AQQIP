# AQQIP User Workflow & Security/Ticker Selection Specification

## 1. User experience objective

AQQIP should feel like an analyst research workspace, not like a collection of developer tools.

The user should be able to answer three questions:

1. **Which company/security do I want to investigate?**
2. **What evidence is currently available?**
3. **What simulated consequences emerge from that evidence?**

## 2. Primary user journey

```text
OPEN AQQIP
   ↓
SELECT / SEARCH SECURITY
   ↓
COMPANY OVERVIEW
   ↓
DATA FRESHNESS CHECK
   ↓
ANALYZE
   ↓
QUANTITATIVE EVIDENCE
   +
QUALITATIVE EVIDENCE
   ↓
SIGNAL BUNDLE
   ↓
SIMULATION OPTIONS
   ↓
MARL SCENARIO
   ↓
RESULTS + EVIDENCE TRACE
```

## 3. How the user chooses a stock

The user should not manually type a CIK.

The primary selection interface is:

```text
Search company name or ticker
```

Example:

```text
Apple
AAPL
Microsoft
MSFT
```

The system resolves the selection to the canonical SEC identity:

```text
Ticker → Company → CIK → supported SEC filer
```

The dashboard should show the resolved identity before analysis starts.

## 4. Search sources

The lookup table should be built from a maintained mapping of:

- ticker,
- company name,
- CIK,
- exchange where available.

The SEC provides company/ticker/CIK lookup resources and filing search functionality. citeturn230039search8turn230039search7

The selected ticker remains the user-facing identifier; CIK is the filing-system identifier.

## 5. Supported-security validation

After selection, the system validates:

```text
1. Ticker exists.
2. CIK exists.
3. Company is an SEC filer supported by v1.
4. At least one relevant 10-K/10-Q can be located.
5. Daily market data exists.
```

If any critical condition fails, the user sees a clear message rather than a half-populated analysis.

## 6. Company analysis landing page

The first screen for a selected ticker should show:

```text
Company
Ticker
CIK
Latest market date
Latest 10-K
Latest 10-Q
Data freshness
Last analysis timestamp
```

Then three primary actions:

```text
[ Refresh Data ]
[ Run Analysis ]
[ Run MARL Scenario ]
```

`Run MARL Scenario` should remain disabled until a valid SignalBundle exists.

## 7. Analyze Now behavior

When the user selects `Run Analysis`:

```text
CHECK CACHE
    ↓
IS REQUIRED DATA FRESH?
  /           \
YES           NO
 |             |
RUN            REFRESH
 |             |
 └──────┬──────┘
        ↓
QUANT ENGINE + QUALITATIVE ENGINE
        ↓
CREATE SIGNAL BUNDLE
        ↓
DISPLAY EVIDENCE
```

The user should never have to manually move outputs between modules.

## 8. Evidence-first result page

The analysis result should be split into two evidence panels.

### Quantitative evidence

Show:

- Beneish M-Score,
- Altman Z-Score,
- key model features,
- calibrated drop probability where available,
- model run ID,
- prediction date,
- target date.

### Qualitative evidence

Show:

- compared filing years,
- cosine similarity,
- semantic shift score,
- added text,
- deleted text,
- shifted text,
- LLM interpretation,
- LLM provider/model.

The LLM summary must be visibly separated from the source filing evidence.

## 9. Evidence quality banner

Every analysis should display a compact evidence state:

```text
QUANT:      AVAILABLE / MISSING / STALE
QUAL:       AVAILABLE / MISSING / STALE
FILING PAIR: AVAILABLE / MISSING
MARKET:     FRESH / CACHED / STALE
```

This is important because a real analyst should know when an inference is based on incomplete evidence.

## 10. How the MARL button works

After analysis, the user sees:

```text
FORENSIC SIGNALS
Quantitative pressure: 0.XX
Disclosure-shift score: 0.XX
Evidence concordance:  0.XX
```

Then:

```text
SCENARIO
○ Baseline
○ Quantitative only
○ Qualitative only
● Combined evidence
○ Stress scenario
```

The dashboard also exposes:

```text
Episode length
Volatility regime
Liquidity regime
Random seed
```

Default values can be provided, but the final run must display what was actually used.

## 11. What the user sees after MARL

The main result should not be a single "crash probability" number.

Show:

```text
Simulated price path
Maximum drawdown
Short-Seller P&L
Retail Herd P&L
Market Maker inventory/risk
Scenario assumptions
Seed
Run ID
```

And a causal explanation generated from logged events:

```text
Quant evidence increased short conviction.
Retail response amplified negative momentum.
Market-maker inventory risk caused liquidity reduction.
These interactions increased simulated drawdown under this scenario.
```

## 12. Evidence trace

Every displayed result should have a route back to its source.

Example:

```text
MARL result
   ↓
SignalBundle hash
   ↓
Quant model run ID + qualitative diff ID
   ↓
Prediction / filing accession
   ↓
Original market / filing evidence
```

This is the core analyst-auditability feature.

## 13. Watchlist mode

The dashboard may provide a watchlist where the user can maintain a small set of securities.

For each ticker show:

```text
Ticker | Latest filing | Quant signal | Qual signal | Freshness | Last run
```

A user can then choose one company for deep analysis.

This is preferable to making users manually enter a ticker every time.

## 14. No automatic "best stock to short" feature

AQQIP should not scan the universe and instruct the user which security to buy or short.

A future ranking view may show research signals for triage, but output should remain framed as:

```text
research priority
anomaly score
scenario risk
```

not:

```text
BUY
SELL
SHORT NOW
```

## 15. Real-world analyst workflow

A realistic usage session is:

### Step 1
Analyst selects a company.

### Step 2
AQQIP resolves its ticker/CIK identity.

### Step 3
System checks whether new 10-K/10-Q data has arrived.

### Step 4
Latest data are ingested or cached data are reused.

### Step 5
Quantitative forensic analysis executes.

### Step 6
Consecutive comparable filings are semantically compared.

### Step 7
AQQIP presents evidence, not a verdict.

### Step 8
Analyst launches a controlled MARL scenario.

### Step 9
AQQIP shows plausible simulated consequences under documented assumptions.

### Step 10
The prediction is persisted with a target date and can later be resolved by the MLOps loop.

## 16. Dashboard safety language

Every output screen must make clear:

> AQQIP is a financial research and simulation tool. Outputs are model-generated signals and simulated scenarios, not investment advice or fraud verdicts.

