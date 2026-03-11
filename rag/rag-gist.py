from dotenv import load_dotenv
import argparse
import logging
import re
import weaviate
from weaviate.client import WeaviateClient
from weaviate.classes.config import Configure, DataType, Property
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaEmbeddings,OllamaLLM
from langchain_weaviate import WeaviateVectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import os

logger = logging.getLogger(__name__)

load_dotenv()

def connect_to_weaviate()-> WeaviateClient:
    ollama_endpoint = os.getenv("OLLAMA_API_ENDPOINT", "")
    return weaviate.connect_to_local(
        headers={"X-Ollama-Api-Endpoint": ollama_endpoint},
    )

def sanitize_metadata_keys(docs: list[Document]) -> list[Document]:
    def to_graphql_name(key: str) -> str:
        sanitized = re.sub(r"[^_0-9A-Za-z]", "_", key)
        if sanitized and sanitized[0].isdigit():
            sanitized = "_" + sanitized
        return sanitized

    for doc in docs:
        doc.metadata = {to_graphql_name(k): v for k, v in doc.metadata.items()}
    return docs

COLLECTION_NAME = "RagGist"

def create_collection_if_missing(client: WeaviateClient):
    if client.collections.exists(COLLECTION_NAME):
        logger.info("Collection '%s' already exists, skipping creation.", COLLECTION_NAME)
        return
    api_endpoint = os.getenv("OLLAMA_API_ENDPOINT")
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT, skip_vectorization=True),
        ],
        vector_config=Configure.Vectors.text2vec_ollama(
            model="embeddinggemma",
            api_endpoint=api_endpoint,
            source_properties=["text"],
        ),
        generative_config=Configure.Generative.ollama(
            model="ministral-3:14b",
            api_endpoint=api_endpoint,
        )
    )
    logger.info("Collection '%s' created.", COLLECTION_NAME)

def delete_collection(client: WeaviateClient):
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        logger.info("Collection '%s' deleted.", COLLECTION_NAME)
    else:
        logger.info("Collection '%s' does not exist, nothing to delete.", COLLECTION_NAME)

def is_collection_populated(client: WeaviateClient) -> bool:
    collection = client.collections.get(COLLECTION_NAME)
    result = collection.aggregate.over_all(total_count=True)
    return (result.total_count or 0) > 0

def load_pdf_chunks(data_dir: str = "data") -> list[Document]:
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logger.error("No PDF files found in data directory.")
        raise FileNotFoundError("No PDF files found in data directory.")

    documents = []
    for file in pdf_files:
        logger.info("Loading %s", file)
        loader = PyPDFLoader(f"{data_dir}/{file}")
        documents.extend(loader.load())

    logger.info("Splitting documents")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    chunks = sanitize_metadata_keys(chunks)
    logger.info("Split into %d chunks.", len(chunks))
    return chunks

def ingest_documents(client: WeaviateClient, embeddings: OllamaEmbeddings):
    if is_collection_populated(client):
        logger.info("Collection already populated, skipping ingestion.")
        return

    logger.info("Loading PDF files")
    chunks = load_pdf_chunks()

    logger.info("Embedding and ingesting documents")
    WeaviateVectorStore.from_documents(
        chunks,
        client=client,
        embedding=embeddings,
        index_name=COLLECTION_NAME,
    )
    logger.info("Ingestion complete.")

def build_vector_store(client: WeaviateClient, embeddings: OllamaEmbeddings) -> WeaviateVectorStore:
    return WeaviateVectorStore(
        client=client,
        index_name=COLLECTION_NAME,
        text_key="text",
        embedding=embeddings,
    )

def build_retriever(db: WeaviateVectorStore, k: int = 3):
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k})

def build_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        "You are a helpful assistant that can answer questions about the following text: {context}"
        "Answer the question: {question}"
    )

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class RagRetriever:
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        prompt_template: ChatPromptTemplate,
        llm: OllamaLLM,
    ):
        self.retriever = retriever
        self.prompt_template = prompt_template
        self.llm = llm

    def retrieval_chain_no_lcel(self, query: str) -> str:
        """Retrieval chain without lcel."""
        docs = self.retriever.invoke(query)
        context = format_docs(docs)
        messages = self.prompt_template.format_messages(context=context, question=query)
        return self.llm.invoke(messages)

    def _retrieval_chain_with_lcel(self)->Runnable:
        retrieval_chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | self.retriever | format_docs 
            )
            | self.prompt_template 
            | self.llm 
            | StrOutputParser()
        )
        return retrieval_chain

    def run_lcel_retrieval(self,query:str):
        retrieval_chain = self._retrieval_chain_with_lcel()
        return retrieval_chain.invoke({"question": query})

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="RAG pipeline with Weaviate and Ollama.")
    parser.add_argument(
        "-r", "--reinitialize",
        action="store_true",
        help="Drop and recreate the Weaviate collection before ingesting documents.",
    )
    args = parser.parse_args()

    ollama_endpoint = os.getenv("OLLAMA_API_ENDPOINT", "")
    embeddings = OllamaEmbeddings(
        model="embeddinggemma",
        base_url=ollama_endpoint,
    )

    llm = OllamaLLM(
        model="ministral-3:14b",
        base_url=ollama_endpoint,
    )

    with connect_to_weaviate() as client:
        if args.reinitialize:
            delete_collection(client)
        create_collection_if_missing(client)
        ingest_documents(client, embeddings)

        db = build_vector_store(client, embeddings)
        retriever = build_retriever(db)
        prompt_template = build_prompt_template()

        rag = RagRetriever(retriever=retriever, prompt_template=prompt_template, llm=llm)
        logger.info("Querying")
        response = rag.retrieval_chain_no_lcel("Co dzieje się jeśli przestępca uzyskał korzyść majątkową?")
        logger.info("Response: %s", response)
        response = rag.run_lcel_retrieval("Co dzieje się jeśli przestępca uzyskał korzyść majątkową?")
        logger.info("Response: %s", response)

if __name__ == "__main__":
    main()