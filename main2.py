import os
import logging
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from groq import Groq
from transformers.agents import Tool, ReactJsonAgent

import nltk
nltk.download('punkt')


# Setup Logging
logger = logging.getLogger("Admino")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ensure DATA directory exists
if not os.path.exists("DATA"):
    os.mkdir("DATA")

# Load documents
logger.info("Loading documents from DATA directory\n")
loader = DirectoryLoader("DATA", glob="*.pdf", show_progress=True)
docs = loader.load()

if not docs:
    logger.error("No documents found in the DATA directory! Please add some PDF files.")
    exit()

# Text splitter
logger.info("Initializing text splitter\n")
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=[".", "\n\n", " ", "\n", ""],
)

# Process and split documents
logger.info("Splitting documents\n")
docs_processed = []
unique_texts = {}
for doc in tqdm(docs, desc="Processing documents"):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

# Embedding model
model_name = "thenlper/gte-small"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name, 
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# FAISS Vector Database
logger.info("Creating Vector DB\n")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embeddings=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)

# Define Retriever Tool
class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Using semantic similarity, retrieves documents from the knowledge base."
    )

    def __init__(self, vectordb, **kwargs):
        self.vectordb = vectordb
        super().__init__(**kwargs)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string"
        docs = self.vectordb.similarity_search(query, k=7)
        return "\n Retrieved documents:\n" + "".join(
            [f"==== Document {i} ====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

# Instantiate Retriever Tool
retriever_tool = RetrieverTool(vectordb)

# Set up Groq API key
os.environ["GROQ_API_KEY"] = "your-groq-api-key"

# Initialize ChatGroq
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.7,
    max_tokens=2048,
)

# OpenAI-like LLM Engine
openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OpenAIEngine:
    def __init__(self, model_name="llama3-groq-70b-8192-tool-use-preview"):
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)
        response = self.client.query(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
            max_tokens=2048,
        )
        return response.choices[0].message.content


llm_engine = OpenAIEngine()

# Create Agent
agent = ReactJsonAgent(
    tools=[retriever_tool],
    llm_engine=llm_engine,
    max_iterations=4,
    verbose=2,
)

# Helper function
def run_agentic_rag(question: str) -> str:
    enhanced_question = f"""Using your knowledge base, retrieve documents with the 'retriever' tool to answer:
    
    {question}
    """
    return agent.run(enhanced_question)

# Main execution
if __name__ == "__main__":
    print("Welcome to the RAG System!")
    print("Please ensure that the DATA folder contains the PDF documents you want to query.")
    
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() == 'exit':
            print("Exiting. Goodbye!")
            break
        
        try:
            response = run_agentic_rag(user_query)
            print("\nResponse:\n", response)
        except Exception as e:
            logger.error(f"An error occurred: {e}")

