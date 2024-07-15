import tiktoken
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding

config_list_openai = [
    {"model": "gpt-4", "api_key": 'sk-proj-KHSb250dMcys1c0VMaUMT3BlbkFJcbC5j9KCUkvWZTgZFdq2'}
]

config_list_groq = [
    {"model": "llama3-70b-8192", "base_url": "https://api.groq.com/openai/v1/", 
     'api_key': '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'},
]
# LLama3 LLM from Lite-LLM Server for Agents #
llm_config_autogen = {
    "seed": 42,  # change the seed for different trials
    "temperature": 0,
    "config_list": [{"model": "litellm", 
                     "base_url": "http://0.0.0.0:4000/", 
                     'api_key': 'ollama'},
    ],
    "timeout": 60000,
}

# Mistral LLM from Ollama for GraphRAG Inference #
llm_config_graphRAG = ChatOpenAI(
    api_key = 'ollama',
    model = "mistral",
    api_type = "openai_chat",
    max_retries=20,
    api_base = 'http://localhost:11434/v1',
)

# Nomic-Text from Ollama for GraphRAG Embedding #
text_embedder = OpenAIEmbedding(api_key='ollama', 
                                api_type="openai_embedding",
                                api_base= "http://localhost:11434/api", 
                                model="nomic-embed-text", 
                                max_retries=20
                                )

token_encoder = tiktoken.get_encoding("cl100k_base")