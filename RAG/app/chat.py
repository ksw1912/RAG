# chat.py

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LangChain이 지원하는 다른 채팅 모델을 사용합니다. 여기서는 Ollama를 사용합니다.
llm = ChatOllama(model="EEVE-Korean-10.8B:latest")

# Prompt 설정
prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Your name is '컴공AI' 너는 항상 반말을 하는 챗봇이야. 다나까나 요 같은 높임말로 절대로 끝내지 마 항상 반말로 친근하게 대답해줘 "
                        "영어로 질문을 받아도 무조건 한글로 답변해줘 한글이 아닌 답변일 때는 다시 생각해서 꼭 한글로 만들어줘 "
                        "모든 답변 시작 끝에는 웃는 이모티콘을 추가해줘 그리고 설명은 최대 3줄까지 설명가능해 3줄이 넘을 것 같으면 요약해서 설명해줘"
                    ),
                    (
                        "system",
                        "When speaking, print a smiling emoticon at the beginning and end of your response."
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

# LangChain 표현식 언어 체인 구문을 사용합니다.
conversation_chain = prompt | llm | StrOutputParser()
