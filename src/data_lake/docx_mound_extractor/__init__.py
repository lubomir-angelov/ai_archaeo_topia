"""ETL pipeline for extracting mound samples from docx files."""

from .loader import DocxLoader
from .models import DocxSample, ExtractionReport, FileStats
from .writer import ArtifactWriter

__all__ = [
    "ArtifactWriter",
    "DocxLoader",
    "DocxSample",
    "ExtractionReport",
    "FileStats",
]
