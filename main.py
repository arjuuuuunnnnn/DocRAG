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
from langchain_groq import ChatGroq
from groq import Groq
from typing import List, Dict
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from huggingface_hub import InferenceClient
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
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
                query,
                k=7,
            )

        return "\n Retrieved documents:\n" + "".join(
                [f"==== Document {str(i)} ====\n" + doc.page_content for i, doc in enumerate(docs)]
            )

# instance of RetrieverTool
retriever_tool = RetrieverTool(vectordb)

os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"

llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.7,
        max_tokens=2048,
    )

openai_role_conversions = {
        MessageRole.TOOL_RESPONSE: MessageRole.USER,
    }

class OpenAIEngine:
    def __init__(self, model_name="llama3-groq-70b-8192-tool-use-preview"):
        self.model_name = model_name
        self.client = Groq(
                api_key=os.getenv("GROQ_API_KEY"),
            )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.query(
                model=self.model_name,
                messages=messages,
                stop=stop_sequences,
                temperature=0.5,
                max_tokens=2048
            )
        return response.choices[0].message.content

llm_engine = OpenAIEngine()



# create agent
agent = ReactJsonAgent(
        tools=[retriever_tool],
        llm_engine=llm_engine,
        max_iterations=4,
        verbose=2
    )


# helper to run the agent
def run_agentic_rag(question:str) -> str:
    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool, give a comprehensive answer to the question below. Respond only to the question asked, response should be concise and relevant to the question. If you cannot find information, do not give up and try calling your retriever again with different arguments! Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries. Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

    Question:
    {question}"""
    
    return agent.run(enhanced_question)

