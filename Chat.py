import os
import json
from dotenv import load_dotenv

# Add OpenAI import
from openai import AzureOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.text_splitter import CharacterTextSplitter
from azure.search.documents.indexes.models import (
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
from tqdm import tqdm
import os
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
import warnings
warnings.filterwarnings("ignore") 
from azure.identity import DefaultAzureCredential


# Get configuration settings 
load_dotenv()
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_openai_api_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
azure_oai_text_deployment = os.getenv("AZURE_OAI_TEXT_DEPLOYMENT")
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")
azure_search_index = os.getenv("AZURE_SEARCH_INDEX")
credential = DefaultAzureCredential()
sc_name = "scoring_profile"

def embedding_func():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_oai_text_deployment,
        api_key=azure_openai_api_key,
        azure_endpoint=azure_oai_endpoint
    )
    return embeddings
def fields_definition(embeddings):
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=len(embeddings.embed_query("Text")),
            vector_search_profile_name="myHnswProfile",
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        # Additional field for filtering on document source
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        # Additional data field for last doc update
        SimpleField(
            name="last_update",
            type=SearchFieldDataType.DateTimeOffset,
            searchable=True,
            filterable=True,
        ),
    ]
    # Adding a custom scoring profile with a freshness function
    
    sc = ScoringProfile(
        name=sc_name,
        text_weights=TextWeights(weights={"content": 5}),
        function_aggregation="sum",
        functions=[
            FreshnessScoringFunction(
                field_name="last_update",
                boost=100,
                parameters=FreshnessScoringParameters(boosting_duration="P2D"),
                interpolation="linear",
            )
        ],
    )
    return fields, sc
def create_vector_store(embeddings, fields, sc):

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_key,
        index_name=azure_search_index,
        embedding_function=embeddings.embed_query,
        fields=fields,
        scoring_profiles=[sc],
        default_scoring_profile=sc_name,
    )
    return vector_store
def create_llm(vector_store):
    azureai_retriever = vector_store.as_retriever(fetch_k=3, fetch_metadata=True)
    # azureai_retriever.invoke("How is Windows OEM revenue growth?")

    llm = AzureChatOpenAI(azure_endpoint=azure_oai_endpoint,
                        api_key=azure_openai_api_key, 
                        api_version="2023-09-01-preview",
                        azure_deployment=azure_oai_deployment,
                        temperature=0.1)
    chat = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=azureai_retriever,
        metadata={"application_type": "question_answering"},
        return_source_documents=True,
    )
    return chat

if __name__ == "__main__":

    embeddings = embedding_func()
    # embedding_function=embeddings.embed_query
    fields, sc = fields_definition(embeddings)
    vector_store = create_vector_store(embeddings, fields, sc)
    chat = create_llm(vector_store)
    query = "who is gregor samsa"
    # query = "who is General Manager, Investor Relations"
    # query = "Activision Blizzard"
    try:
        response = chat.invoke({"query": query})
        # Accessing the list of documents
        documents = response['source_documents']

        # Extracting metadata from each document
        response_metadata = set()
        for document in documents:
            response_metadata.add(document.metadata['source'])

        print(f'Query: {response["query"]} : , Response: {response["result"]}, Source Documents: {response_metadata}')
    except Exception as e:
        print(f"An error occurred: {e}")
