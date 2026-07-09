"""Custom application exceptions."""


class BidaiError(Exception):
    """Base exception for the bidai application."""


class PDFError(BidaiError):
    """Base exception for PDF-related errors."""


class PDFNotFoundError(PDFError):
    """Raised when the PDF file path does not exist."""


class PDFInvalidFileError(PDFError):
    """Raised when the input file is not a valid PDF."""


class PDFReadError(PDFError):
    """Raised when a PDF cannot be opened or read."""


class PDFEncryptedError(PDFError):
    """Raised when a PDF is encrypted and cannot be read without a password."""


class PDFExtractionError(PDFError):
    """Raised when text extraction from a PDF fails."""


class ChunkingError(BidaiError):
    """Base exception for document chunking errors."""


class InvalidChunkConfigError(ChunkingError):
    """Raised when chunk size or overlap settings are invalid."""


class EmptyExtractedDocumentError(ChunkingError):
    """Raised when an extracted document has no pages."""


class IndexingError(BidaiError):
    """Base exception for vector indexing errors."""


class EmptyDocumentListError(IndexingError):
    """Raised when indexing is called with an empty document list."""


class MissingDocumentMetadataError(IndexingError):
    """Raised when a document is missing required metadata fields."""


class InvalidDocumentIdError(IndexingError):
    """Raised when a document ID is missing or invalid."""


class VectorStoreError(IndexingError):
    """Raised when Chroma vector store initialization fails."""


class IndexingFailureError(IndexingError):
    """Raised when adding documents to the vector store fails."""


class MissingOpenAIAPIKeyError(IndexingError):
    """Raised when OpenAI embeddings are requested without an API key."""
