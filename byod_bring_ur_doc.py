from embedding_model import EmbeddingModel
from document_loader import DocumentLoader
from text_splitter import TextSplitter
from vector_space import VectorSpace
from merger_retriever import MergerRetriever
import os

# Part 1: Get the Embedding Model
model_name = "BAAI/bge-large-en"
model = EmbeddingModel(model_name)

# Part 4: Load Vector vector_space
pdf_files = ["data/000885.pdf", "data/000945.pdf"]
vector_spaces = []

for file_path in pdf_files:
    vector_space = VectorSpace([], model, f"vector_space/{os.path.basename(file_path)}_chroma_cosine")
    vector_spaces.append(vector_space.load_vector_space())

# Part 5: Init Merge Retriever
retrievers = [vs.as_retriever(search_type="similarity", search_kwargs={"k": 3}) for vs in vector_spaces]
lotr = MergerRetriever(retrievers)

query = "marine protected areas are being designated according which approaches?"
for chunks in lotr.get_relevant_documents(query):
    print(chunks.page_content)
