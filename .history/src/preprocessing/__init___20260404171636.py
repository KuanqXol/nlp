"""Preprocessing package: NER, entity linking, relation extraction."""

from src.preprocessing.ner import (
    VietnameseNER,
    get_entities_by_type,
    ner_with_checkpoint,
    resolve_coreference,
)
from src.preprocessing.entity_linking import EntityLinker
from src.preprocessing.relation_extraction import RelationExtractor

__all__ = [
    "VietnameseNER",
    "get_entities_by_type",
    "ner_with_checkpoint",
    "resolve_coreference",
    "EntityLinker",
    "RelationExtractor",
]
