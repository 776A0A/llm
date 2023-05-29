from dotenv import load_dotenv
from langchain import OpenAI
from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.node_parser import SimpleNodeParser

# import logging, sys

load_dotenv()

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = SimpleDirectoryReader("data").load_data()

parser = SimpleNodeParser()

nodes = parser.get_nodes_from_documents(documents=documents)

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt3.5-turbo"))
max_input_size = 4096
num_output = 256
max_chunk_overlap = 20

prompt_helper = PromptHelper(
    max_input_size=max_input_size,
    num_output=num_output,
    max_chunk_overlap=max_chunk_overlap,
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

# index = GPTVectorStoreIndex.from_documents(documents=documents)
# index = GPTVectorStoreIndex(nodes=nodes)
index = GPTVectorStoreIndex.from_documents(
    documents=documents, service_context=service_context
)

query_engine = index.as_query_engine(verbose=True)

response = query_engine.query("What did the author do growing up?")

print(response)
