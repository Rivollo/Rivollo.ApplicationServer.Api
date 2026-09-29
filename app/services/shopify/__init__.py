"""Shopify integration services (docs/shopify-integration/spec.md, ADR-016).

An isolated module: it calls ProductService, ModelVariantGenerationService and
the generation gate, and never modifies them.
"""
