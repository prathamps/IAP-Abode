from dotenv import load_dotenv
import os
load_dotenv()

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_parse import LlamaParse
from llama_index.core import  VectorStoreIndex, SummaryIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors.utils import get_selector_from_llm

api_key = os.getenv("GOOGLE_API_KEY")
llama_parser_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

Settings.llm = GoogleGenAI(model="gemini-2.5-flash-lite", api_key=api_key)
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001", api_key=api_key)


parser = LlamaParse(
    api_key= llama_parser_api_key,
    result_type="markdown",
)
documents = parser.load_data("data/clean_code.pdf")

vectorIndex=VectorStoreIndex.from_documents(documents)
summaryIndex=SummaryIndex.from_documents(documents)

vectorEngine=vectorIndex.as_query_engine(similarity_top_k=5)
summaryEngine=summaryIndex.as_query_engine(
    responses_model='tree_summarize',
    use_async=True
)

vectorTool = QueryEngineTool.from_defaults(
    query_engine=vectorEngine,
    description=(
        "Use this for questions that need specific facts, quotes, definitions, "
        "or details from the document."
    ),
)
summaryTool = QueryEngineTool.from_defaults(
    query_engine=summaryEngine,
    description=(
        "Use this for summarization questions, high-level overviews, or "
        "when the user asks to summarize a chapter/section/document."
    ),
)

tools = [vectorTool, summaryTool]
selector=get_selector_from_llm(Settings.llm)

routerEngine=RouterQueryEngine(
    selector=selector,
    query_engine_tools=tools,
    verbose=True
)

while True:
    response=routerEngine.query(input("enter your query:"))
    print(response)

