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
