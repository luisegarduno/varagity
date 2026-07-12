"""Runtime settings override layer — Phase 8 stub (spec_v2 §4.7).

The persisted override layer (an ``app_settings`` table merged over the
env :class:`~varagity.config.Settings`, cache-cleared on ``PATCH
/api/settings``) lands with its GUI in Phase 8, reusing the
``pinned_eval_settings`` mechanism (plan decision #9). Until then the API
exposes only static capabilities (``GET /api/config``) and honors
per-request ``overrides`` on ``POST /api/chat``; this module reserves the
seam so Phase 8 adds a layer, not a rewrite.
"""
