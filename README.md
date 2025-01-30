# DocRAG : Talk with your PDFs
An intelligent document interaction system that lets you have natural conversations with your PDF documents using state-of-the-art large language models and semantic search.


## Features
 - Command line interface
 - Natural conversation with your PDFs
 - Language understanding with Groq's LLama 3 70B model


## Setup
1. clone
```bash
git clone https://github.com/DocRAG.git
cd DocRAG
```
2. Add your Docs
Create a `DATA` folder in the project root
Drop your PDF files into the `DATA` folder

3. Set your API key
Create a `.env` file in the project root
Add your Groq API key to the `.env` file
```bash
GROQ_API_KEY="your api key"
```

4. Run DocRAG
```bash
python main.py
```


## Advanced Config
Change the behaviour of DocRAG by adjusting
```bash
# Document Processing
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

# LLM Settings
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# Agent Config
MAX_ITERATIONS = 4
```


