from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    GPTListIndex,
    GPTVectorStoreIndex,
    LangchainEmbedding,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    download_loader,
)
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaToolkit,
    create_llama_chat_agent,
)
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

load_dotenv()


UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
loader = UnstructuredReader()

years = [year for year in range(2019, 2023)]


doc_set = {}
all_docs = []
for year in years:
    docs = loader.load_data(
        file=Path(f"./data/UBER/UBER_{year}.html"), split_documents=False
    )
    for doc in docs:
        doc.extra_info = {"year": year}
    doc_set[year] = docs
    all_docs.extend(docs)

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=512)
llm_predictor = LLMPredictor(llm=llm)

service_context = ServiceContext.from_defaults(
    embed_model=embed_model, llm_predictor=llm_predictor, chunk_size_limit=512
)

index_set = {}

for year in years:
    storage_context = StorageContext.from_defaults()
    cur_index = GPTVectorStoreIndex.from_documents(
        doc_set[year], service_context=service_context, storage_context=storage_context
    )
    index_set[year] = cur_index

index_summaries = [f"UBER 10-k Filing for {year} fiscal year" for year in years]

graph_storage_context = StorageContext.from_defaults()

graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[year] for year in years],
    index_summaries=index_summaries,
    service_context=service_context,
    storage_context=graph_storage_context,
)


decompose_transform = DecomposeQueryTransform(llm_predictor=llm_predictor, verbose=True)

index_configs = []

for year, index in index_set.items():
    query_engine = TransformQueryEngine(
        query_engine=index.as_query_engine(similarity_top_k=3),
        query_transform=decompose_transform,
        transform_extra_info={"index_summary": index.index_struct.summary},
    )
    tool_config = IndexToolConfig(
        query_engine=query_engine,
        name=f"Vector Index {year}",
        description=f"useful for when you want to answer queries about the {year} SEC 10-K for Uber",
        tool_kwargs={"return_direct": True},
    )
    index_configs.append(tool_config)

graph_query_engine = graph.root_index.as_query_engine(
    response_mode="tree_summarize", verbose=True
)

graph_config = IndexToolConfig(
    query_engine=graph_query_engine,
    name="Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for Uber.",
    tool_kwargs={"return_direct": True},
)

toolkit = LlamaToolkit(index_configs=index_configs + [graph_config])

memory = ConversationBufferMemory(memory_key="chat_history")

agent_chain = create_llama_chat_agent(toolkit, llm, memory=memory, verbose=True)

while True:
    text_input = input("User: ")
    response = agent_chain.run(input=text_input)
    print(f"Agent: {response}")
