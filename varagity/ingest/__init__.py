"""Ingestion pipeline: discovery, parsing, and the loading orchestrator.

The flow (spec §9): ``discover → parse → chunk → contextualize → embed →
store``. Contextualization is the LLM situating-blurb step when
``settings.CONTEXTUALIZE`` is on, and the identity path
(``contextualized_content = content``) when off — the non-contextual
baseline (plan decision #2).
"""
