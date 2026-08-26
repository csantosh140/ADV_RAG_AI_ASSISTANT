"""Custom domain exceptions for Advanced RAG AI System."""


class RAGException(Exception):
    """Base exception for all RAG system errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentParsingError(RAGException):
    """Raised when parsing a document fails."""
    pass


class UnsupportedFileTypeError(RAGException):
    """Raised when an unsupported file format is uploaded."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector storage operations fail."""
    pass


class RetrievalError(RAGException):
    """Raised when chunk retrieval encounters an error."""
    pass


class LLMInferenceError(RAGException):
    """Raised when LLM invocation fails."""
    pass


class HallucinationDetectedError(RAGException):
    """Raised when answer violates groundedness constraints."""
    pass
