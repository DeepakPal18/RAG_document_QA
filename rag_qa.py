import os
import openai
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

# Set OpenAI API key
openai_api_key = "sk-1123344" # api key
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = os.getenv("OPENAI_API_KEY")


# Initialize an OpenAI language model (llm) with specific configurations
from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=500)


# Check if the indexes in storage already exists
PERSIST_DIR = r"E:\GenAI projects\rag_assignment\Indexes\data"
if not os.path.exists(PERSIST_DIR):
    # if not present then load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # if present then load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Initialize the retriever and postprocessor(query engine)
# Retrievers -  are used in query engines to retrieve the most similar responses with respect to your queries.
# postprocessor - processing the retrieved nodes from theretrieval phase.
retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.75)

# Create the query engine
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

# Function to query the engine and include source documents, run this if you want to see the text and referred docs only in rag_qa file output
# def query_engine_function(query):
#     response = query_engine.query(query)
#     cited_documents = [node.node.ref_doc_id for node in response.source_nodes]
#     result = f"Answer: {response.response}\n\nReferred Documents:\n"
#     for doc in cited_documents:
#         result += f"- {doc}\n"
#     return result

# Function to query the engine (returning object type)
# we will extract response text and metadata from this response object in streamlit function
def query_engine_function(query):
    response = query_engine.query(query)
    return response


# Main loop to interact with the user
def main():
    print("Welcome to the Document Query System. Type your question or 'exit' to quit.")
    while True:
        query = input("Enter your question or Type 'exit' to quit..? : ")
        if query.lower() == 'exit':
            print("Exiting the query system. Goodbye!")
            break
        response = query_engine_function(query)
        print(response)

if __name__ == "__main__":
    main()