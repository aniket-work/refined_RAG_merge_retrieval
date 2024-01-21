from typing import List
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

def rerank_documents(documents: List[Document]) -> List[Document]:
    """
    Re-rank a list of documents using the LongContextReorder algorithm.

    Args:
    - documents (List[Document]): List of documents to be re-ranked.

    Returns:
    - List[Document]: Re-ranked list of documents.
    """
    # Example: Using LongContextReorder for re-ranking
    reorder_algorithm = LongContextReorder()
    reordered_documents = reorder_algorithm.transform_documents(documents)

    return reordered_documents

# Example Usage:
if __name__ == "__main__":
    # Assuming you have a list of documents to be re-ranked
    input_documents = [
        Document(id=1, text="Document 1 content", metadata={"score": 0.8}),
        Document(id=2, text="Document 2 content", metadata={"score": 0.6}),
        # Add more documents as needed
    ]

    # Re-rank the documents using the specified algorithm
    output_documents = rerank_documents(input_documents)

    # Display the re-ranked documents
    for document in output_documents:
        print(f"Document {document.id}: {document.text} (Score: {document.metadata['score']})")
