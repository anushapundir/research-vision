"""
AI Module
Integration with Google Gemini for document understanding and summarization.
"""

from .gemini_client import GeminiClient, generate_summary, analyze_document_structure
from .summary_pipeline import summarize_research_paper, extract_key_findings, generate_abstract

__all__ = [
    'GeminiClient',
    'generate_summary',
    'analyze_document_structure',
    'summarize_research_paper',
    'extract_key_findings',
    'generate_abstract'
]
