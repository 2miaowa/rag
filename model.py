
import os
from typing import List
#from fastapi import FastAPI
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
import os
from glob import glob


def rag_chain_model():
    # 1. Load Retriever
    
    # 指定文件夹路径
    folder_path = 'data'

    # 获取文件夹下所有的文件路径
    file_paths = glob(os.path.join(folder_path, '*'))

    docs = []
    # 按照页面切割
    for file_path in file_paths:
        print(file_path)
        loader = PyMuPDFLoader(file_path)
        pages = loader.load_and_split()
        docs.extend(pages)
    # 每页按照chunk_size切割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OllamaEmbeddings(model="mofanke/acge_text_embedding")
    )
    retriever = vectorstore.as_retriever(k=4)


    # 2. Create Tools
    retriever_tool = create_retriever_tool(
        retriever,
        "中国政府网",
        "Search for information about 中国政府网. For any questions about 中国政府网, you must use this tool!",
    )
    os.environ["TAVILY_API_KEY"] = ""
    search = TavilySearchResults()
    tools = [retriever_tool, search]

    # 3. Create Agent
    template = """基于 context 回答 question :

    {context}

    Question: {question}
    用中文输出回答


    """
    prompt = ChatPromptTemplate.from_template(template)

    os.environ["DASHSCOPE_API_KEY"] = ""
    from langchain_community.chat_models.tongyi import ChatTongyi

    llm = ChatTongyi(model="qwen-plus")


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print('*'*30)
    print(rag_chain)
    print('*'*30)
    return rag_chain
