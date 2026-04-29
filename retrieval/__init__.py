"""
Retrieval package – document construction, cross-source linking,
hybrid retrieval, and local fallback.
"""
from .document_builder import (
    build_all_documents,
    build_ctd_docs,
    build_metagenome_docs,
    build_sst_docs,
    documents_to_dataframe,
    documents_to_jsonl,
)
from .cross_source_linker import build_cross_source_links
