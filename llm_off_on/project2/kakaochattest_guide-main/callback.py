from dto import ChatbotRequest
import requests
import time
import logging
import openai
import os
from langchain.chains import LLMChain, SequentialChain
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

# 환경 변수 처리 필요!
openai.api_key = os.environ.get("OPENAI_API_KEY")
SYSTEM_MSG = """You are a chatbot that informs you of the features supported by KakaoTalk. Your user will be Korean, so you need to communicate in Korean. 
            First of all, you need to ask the user what he or she has any questions about KakaoTalk."""
logger = logging.getLogger("Callback")
llm = ChatOpenAI(temperature=0.1, max_tokens=300, model="gpt-3.5-turbo-16k")


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()
    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path),
        ),
        output_key=output_key,
        verbose=True,
    )


def callback_handler(request: ChatbotRequest) -> dict:
    kakao_talk_info_chain = create_chain(llm, "./project_data_카카오싱크.txt", "kakao_talk_info")

    question = request.userRequest.utterance
    system_message_prompt = SystemMessage(content=SYSTEM_MSG)
    human_message = """reference: {kakao_talk_info} question: {question} \n Please answer the questions about the Kakao Talk above."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_message)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    answer_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        output_key="answer",
        verbose=True,
    )

    preprocess_chain = SequentialChain(
        chains=[
            kakao_talk_info_chain,
            answer_chain,
        ],
        input_variables=["question"],
        output_variables=["kakao_talk_info"],
        verbose=True,
    )

    context = dict(question=question)
    context = preprocess_chain(context)
    context["question"] = question
    context = answer_chain(context)
    output_text = context["answer"]

    while len(output_text) > 1000:
        output_text = output_text[:1000]
        send_kakao_talk_response(output_text, request.userRequest.callbackUrl)
        output_text = output_text[1000:]
    send_kakao_talk_response(output_text, request.userRequest.callbackUrl)

    # focus


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
