import json
from vector_space import VectorSpace
from merger_retriever import MergerRetrieverHandler
from embedding_model import EmbeddingModelFactory
import os

def load_configuration(config_path: str) -> dict:
    """
    Load configuration from a JSON file.

    Parameters:
    - config_path (str): Path to the JSON configuration file.

    Returns:
    - dict: Configuration dictionary loaded from the JSON file.
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return config

def load_pdf_files(pdf_folder: str) -> list:
    """
    Load PDF files from a specified folder.

    Parameters:
    - pdf_folder (str): Path to the folder containing PDF files.

    Returns:
    - list: List of paths to PDF files in the specified folder.
    """
    pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith(".pdf")]
    return pdf_files

def load_vector_spaces(pdf_files: list, model) -> list:
    """
    Load VectorSpace objects for each PDF file.

    Parameters:
    - pdf_files (list): List of paths to PDF files.
    - model: model from EmbeddingModelFactory.

    Returns:
    - list: List of initialized VectorSpace objects.
    """
    vector_spaces = []

    for file_path in pdf_files:
        model = model
        vector_space = VectorSpace([], model, f"vector_space/{os.path.basename(file_path)}_chroma_cosine")
        vector_spaces.append(vector_space.load_vector_space())

    return vector_spaces

def initialize_merger_retriever(vector_spaces: list) -> MergerRetrieverHandler:
    """
    Initialize the MergerRetrieverHandler with VectorSpace objects.

    Parameters:
    - vector_spaces (list): List of initialized VectorSpace objects.

    Returns:
    - MergerRetrieverHandler: Initialized MergerRetrieverHandler.
    """
    retrievers = [vs.as_retriever(search_type="similarity", search_kwargs={"k": 3}) for vs in vector_spaces]
    return MergerRetrieverHandler(retrievers)

def load_queries(queries_path: str) -> list:
    """
    Load queries from a JSON file.

    Parameters:
    - queries_path (str): Path to the JSON file containing queries.

    Returns:
    - list: List of queries.
    """
    with open(queries_path, "r") as queries_file:
        queries = json.load(queries_file)
    return queries

def main():
    # Part 1: Load configuration from JSON file
    config_path = "config/config.json"
    config = load_configuration(config_path)

    # Part 2: Initialize embedding model factory
    factory = EmbeddingModelFactory(default_model=config.get("embeddingModel", "BAAI/bge-large-en"))
    model_custom = factory.create_model(model_name="BAAI/bge-large-en", model_kwargs = {'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': False})

    # Part 3: Load PDF files from "byod" folder
    pdf_folder = "byod"
    pdf_files = load_pdf_files(pdf_folder)

    # Part 4: Load Vector vector_space
    vector_spaces = load_vector_spaces(pdf_files, model_custom)

    # Part 5: Initialize Merge Retriever
    lotr = initialize_merger_retriever(vector_spaces)

    # Part 6: Load queries and run them
    queries_path = "config/queries.json"
    queries = load_queries(queries_path)

    for query in queries:
        print(f"\n------------------------------------")
        print(f"\nQuery: {query}")
        print(f"\n------------------------------------")
        for chunks in lotr.get_relevant_documents(query):
            print(chunks.page_content)
        print(f"\n------------------------------------")

if __name__ == "__main__":
    main()
