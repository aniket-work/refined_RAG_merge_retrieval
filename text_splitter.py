from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextSplitter:
    """
    A class for splitting documents into chunks using a Recursive Character Text Splitter.

    Attributes:
    - chunk_size (int): The size of each text chunk.
    - chunk_overlap (int): The overlap between consecutive text chunks.

    Methods:
    - split_documents(documents): Split a list of documents into text chunks using Recursive Character Text Splitter.
    """

    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initialize the TextSplitter.

        Args:
        - chunk_size (int): The size of each text chunk.
        - chunk_overlap (int): The overlap between consecutive text chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents):
        """
        Split a list of documents into text chunks using Recursive Character Text Splitter.

        Args:
        - documents (list): List of documents to be split.

        Returns:
        - list: List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)
