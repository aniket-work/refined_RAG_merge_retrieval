import json
from embedding_model import EmbeddingModel
from vector_space import VectorSpace
from merger_retriever import MergerRetrieverHandler

# Part 1: Get the Embedding Model
model_name = "BAAI/bge-large-en"
model = EmbeddingModel(model_name)

# Part 4: Load Vector vector_space
vector_spaces_config = "vector_spaces.json"

with open(vector_spaces_config, 'r') as file:
    vector_spaces_data = json.load(file)

vector_spaces = []

for space_data in vector_spaces_data["vector_spaces"]:
    vector_space_name = space_data["name"]
    vector_space = VectorSpace([], model, f"vector_space/{vector_space_name}")
    vector_spaces.append(vector_space.load_vector_space())

# Part 5: Init Merge Retriever
retrievers = [vs.as_retriever(search_type="similarity", search_kwargs={"k": 3}) for vs in vector_spaces]
lotr = MergerRetrieverHandler(retrievers)

query = "marine protected areas are being designated according which approaches?"
for chunks in lotr.get_relevant_documents(query):
    print(chunks.page_content)
