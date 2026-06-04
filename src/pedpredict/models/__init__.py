"""Model components: vit, motion_encoder, cross_attention, ensemble, ablations + typed registry (P2).

Kept import-light on purpose: ``config.loader`` imports the torch-free ``models.geometry`` for validation,
so eager submodule imports here would create a ``config`` <-> ``models`` circular import. Import the
concrete symbols from their modules (e.g. ``from pedpredict.models.registry import build_model``).
"""
