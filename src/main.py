from dotenv import load_dotenv
load_dotenv()

import os
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

api_key = os.getenv("GOOGLE_API_KEY")

Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001", api_key=api_key)

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("where is paris?")
print(response)