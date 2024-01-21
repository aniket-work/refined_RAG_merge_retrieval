from langchain.retrievers.merger_retriever import MergerRetriever

class MergerRetrieverHandler:
    """
    A handler class for managing a MergerRetriever instance.

    Attributes:
    - retrievers (list): List of retriever instances to be used by the MergerRetriever.
    - merger (MergerRetriever): An instance of MergerRetriever for retrieving relevant documents.

    Methods:
    - get_relevant_documents(query): Retrieve relevant documents based on the provided query.
    """

    def __init__(self, retrievers):
        """
        Initialize the MergerRetrieverHandler.

        Args:
        - retrievers (list): List of retriever instances to be used by the MergerRetriever.
        """
        self.retrievers = retrievers
        self.merger = MergerRetriever(retrievers=self.retrievers)

    def get_relevant_documents(self, query):
        """
        Retrieve relevant documents based on the provided query.

        Args:
        - query (str): The query for which relevant documents are to be retrieved.

        Returns:
        - list: List of relevant documents.
        """
        return self.merger.get_relevant_documents(query)
