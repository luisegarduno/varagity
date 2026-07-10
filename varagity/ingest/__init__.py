"""Ingestion pipeline: discovery, parsing, and the loading orchestrator.

The flow (spec §9): ``discover → parse → chunk → contextualize → embed →
store``. Phase 3 wires everything except contextualization, which is the
identity step until Phase 5 (plan decision #1).
"""
