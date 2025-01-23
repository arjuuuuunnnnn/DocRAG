import pandas as pd
import datasets
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers.agents import Tool, HfEngine, ReactJsonAgent
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import os


logger = logging.getLogger("Admino")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


os.mkdir("DATA")


loader = DirectoryLoader("DATA", glob="*.pdf", show_progress=True)
docs = loader.load()

# text splitter
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=[".", "\n\n", " ", "\n", ""],
    )

# docs splitting
logger.info("Splitting documents\n")
docs_processed = []
unique_texts = {}
for doc in tqdm(docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

# embeddings
model_name = "thenlper/gte-small"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# vector db
logger.info("Creating Vector DB\n")
vectordb = FAISS.from_documents(
        documents=docs_processed,
        embeddings=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )


# RAG
class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
            "query": {
                "type": "text",
                "description": "The query to perform. This should be semantically close to the target documents. Use the affirmative form ranther then a question.",
                }
        }
    output_type = "text"

    def __init__(self, vectordb, **kwargs):
        self.vectordb = vectordb
        super().__init__(**kwargs)

    def forward(self, query:str) -> str:
