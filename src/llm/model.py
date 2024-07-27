from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from groq import Groq


LLAMA_8B_LLM = ChatGroq(model_name="llama3-8b-8192")  # 8k context
LLAMA_70B_LLM = ChatGroq(model_name="llama3-70b-8192")  # 8k context
MIXTRAL_LLM = ChatGroq(model_name="mixtral-8x7b-32768")  # 32k context

GEMINI_FLASH_LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # 1m context

# multimodal model
MULTIMODAL_LLM = GEMINI_FLASH_LLM

# embedding model
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# transcript api
GROQ_CLIENT = Groq()
