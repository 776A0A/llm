import os
import platform

import faiss
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index import (
    GPTVectorStoreIndex,
    LangchainEmbedding,
    LLMPredictor,
    PromptHelper,
    QuestionAnswerPrompt,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore


if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    print('Set "KMP_DUPLICATE_LIB_OK" to "True"')


load_dotenv()


text_splitter = CharacterTextSplitter(
    separator="\n\n", chunk_size=100, chunk_overlap=20
)
parser = SimpleNodeParser(text_splitter=text_splitter)

documents = SimpleDirectoryReader("data/faq").load_data()
nodes = parser.get_nodes_from_documents(documents=documents)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))

max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(
    max_chunk_overlap=max_chunk_overlap,
    num_output=num_output,
    max_input_size=max_input_size,
)


embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
)

service_context = ServiceContext.from_defaults(
    embed_model=embed_model, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

dimension = 768
faiss_index = faiss.IndexFlatIP(dimension)

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = GPTVectorStoreIndex.from_documents(
    documents=documents,
    service_context=service_context,
    storage_context=storage_context,
)


QA_PROMPT_TMPL = "{context_str}" "\n\n" "根据以上信息，请回答下面的问题：\n" "Q: {query_str}\n"

qa_prompt = QuestionAnswerPrompt(QA_PROMPT_TMPL)

query_engine = index.as_query_engine(
    retriever_model="embedding", verbose=True, text_qa_template=qa_prompt
)

response = query_engine.query("请问你们海南能发货吗？")


print(response)
