from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
    """
    A class for loading documents from a specified file path, particularly designed for PDF files.

    Attributes:
    - file_path (str): The path to the document file.

    Methods:
    - load_pdf_documents(): Load documents from the specified PDF file.
    """

    def __init__(self, file_path):
        """
        Initialize the DocumentLoader.

        Args:
        - file_path (str): The path to the document file.
        """
        self.file_path = file_path

    def load_pdf_documents(self):
        """
        Load documents from the specified PDF file.

        Returns:
        - list: List of loaded documents.
        """
        loader = PyPDFLoader(self.file_path)
        return loader.load()
