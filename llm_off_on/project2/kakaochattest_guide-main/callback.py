from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.tools import Tool

from dto import ChatbotRequest
import requests
import time
import logging
import openai
import os
from langchain.chains import LLMChain
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.document_loaders import TextLoader
import re

openai.api_key = os.environ.get("OPENAI_API_KEY")
KAKAO_SYNC_DATA_PATH = "./project_data_카카오싱크.txt"
QUESTION_PARSER_PROMPT_PATH = "./question_parser_prompt.txt"
logger = logging.getLogger("Callback")
llm = ChatOpenAI(temperature=0.1, max_tokens=300, model="gpt-3.5-turbo-16k")


def create_chain(llm, template_path, output_parser):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path),
        ),
        verbose=True,
        output_parser=output_parser
    )


def callback_handler(request: ChatbotRequest) -> dict:
    question_chain = create_chain(
        llm=llm,
        template_path=QUESTION_PARSER_PROMPT_PATH,
        output_parser=QuestionParser()
    )

    result = question_chain.run(request.userRequest.utterance)

    functions = [
        {
            "name": "search_db",
            "func": lambda x: search_db(result),
            "description": "Function to search extra information related to query from the database of Kakao sync"
        }
    ]

    tools = [
        Tool(
            **func
        ) for func in functions
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

    output_text = agent.run(result)

    send_kakao_talk_response(output_text, request.userRequest.callbackUrl)


def search_db(query: str):
    def query_db(query: str, use_retriever: bool = False) -> list[str]:
        if use_retriever:
            docs = retriever.get_relevant_documents(query)
        else:
            docs = db.similarity_search(query)

        str_docs = [doc.page_content for doc in docs]
        return str_docs

    loader = TextLoader(KAKAO_SYNC_DATA_PATH)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(
        texts,
        OpenAIEmbeddings(),
        collection_name="kakao_sync",
    )

    retriever = db.as_retriever()

    query_result = query_db(
        query=query,
        use_retriever=True,
    )

    search_results = []
    for document in query_result:
        search_results.append(
            {
                "content": document.split(':')[1]
            }
        )

    return search_results


def send_kakao_talk_response(output_text: str, url: str):
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }

    time.sleep(1.0)

    if url:
        resp = requests.post(url=url, json=payload)
        print(resp.status_code)


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()
    return prompt_template


class QuestionParser(BaseOutputParser):
    def parse(self, output: str) -> str:
        result = re.match(r"Detected: (.*)", output)
        return result.group()

    def get_format_instructions(self) -> str:
        return "Your response should be in following format.\nDetected: <detected abbreviates several words>"
