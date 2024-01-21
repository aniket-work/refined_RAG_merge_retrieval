from langchain_community.vectorstores import Chroma

class VectorSpace:
    """
    A class for managing vector spaces using the Chroma vector store.

    Attributes:
    - texts (list): List of texts to be included in the vector space.
    - embedding_model: The embedding model used to generate vectors.
    - persist_directory (str): The directory to persist the vector space.

    Methods:
    - create_chroma_vector_space(): Create a Chroma vector space from the provided texts and embedding model.
    - load_chroma_vector_space(): Load a Chroma vector space from the specified persist directory.
    """

    def __init__(self, texts, embedding_model, persist_directory):
        """
        Initialize the VectorSpace.

        Args:
        - texts (list): List of texts to be included in the vector space.
        - embedding_model: The embedding model used to generate vectors.
        - persist_directory (str): The directory to persist the vector space.
        """
        self.texts = texts
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory

    def create_chroma_vector_space(self):
        """
        Create a Chroma vector space from the provided texts and embedding model.

        Returns:
        - Chroma: The created Chroma vector space.
        """
        return Chroma.from_documents(
            self.texts,
            self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=self.persist_directory
        )

    def load_chroma_vector_space(self):
        """
        Load a Chroma vector space from the specified persist directory.

        Returns:
        - Chroma: The loaded Chroma vector space.
        """
        return Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
