"""
Pipeline Module
Orchestrates the end-to-end document processing pipeline.
"""

from .end_to_end import process_document, DocumentPipeline

__all__ = ['process_document', 'DocumentPipeline']
