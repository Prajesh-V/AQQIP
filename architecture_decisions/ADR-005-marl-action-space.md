# ADR-005: MARL Discrete Action Spaces

## Status
Accepted

## Context
The MARL environment simulates three market participant types (Short Seller, Retail Herd, Market Maker). The action space design affects stability, interpretability, and debugging difficulty.

## Decision
Use discrete action spaces for all three agents in v1:

**Short Seller:** 0=Hold, 1=Increase Short, 2=Reduce Short/Cover
**Retail Herd:** 0=Hold, 1=Buy, 2=Sell
**Market Maker:** 0=Normal Liquidity, 1=Increase Liquidity, 2=Reduce Liquidity

## Alternatives Considered
- Continuous action spaces → rejected for v1 (harder to debug, more policy instability, not needed until rule-based agents are proven correct)
- Larger discrete action set (5+ actions per agent) → deferred (adds complexity without demonstrated benefit in initial experiments)

## Rationale
Discrete 3-action spaces make the environment deterministically testable with rule-based agents before introducing learned policies. This separation prevents policy-learning instability from hiding environment bugs.

## Consequences
- Position sizing is implicitly fixed per action (configurable step size)
- The transition model must map discrete actions to continuous market effects
- Rule-based agents are trivially implementable as threshold policies
- Continuous-action extension is possible in a future version without changing the environment interface
