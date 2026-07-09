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


class MissingOpenAIAPIKeyError(BidaiError):
    """Raised when OpenAI API access is requested without an API key."""


class RetrievalError(BidaiError):
    """Base exception for document retrieval errors."""


class EmptyQuestionError(RetrievalError):
    """Raised when a question is empty or whitespace-only."""


class InvalidTopKError(RetrievalError):
    """Raised when top_k is not a positive integer."""


class CollectionNotFoundError(RetrievalError):
    """Raised when no indexed collection exists for the given document ID."""


class NoRetrievedDocumentsError(RetrievalError):
    """Raised when retrieval returns no documents for a question."""


class RAGError(BidaiError):
    """Base exception for RAG answer-generation errors."""


class ChatAPIError(RAGError):
    """Raised when the OpenAI Chat API call fails."""


class GraphError(BidaiError):
    """Base exception for LangGraph workflow failures."""


class GraphInvocationError(GraphError):
    """Raised when the compiled RAG graph invocation fails unexpectedly."""
