import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
  model="gpt-3.5-turbo",
  temperature=0.5,
  api_key=api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a travel guide specializing in Brazil's locations. You are called Mr. Passeio"),
    ("placeholder", "{history}"),
    ("human", "{query}")
])

chain = prompt | model | StrOutputParser()

memory = {}
session = "langchain_class"

def sessionHistory(session: str):
  if session not in memory:
    memory[session] = InMemoryChatMessageHistory()
  return memory[session]

questions = [
  "I want to visit a place in Brazil famous for it's beaches and culture. Can you suggest something?",
  "What is the best time to visit this place?",
]

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=sessionHistory
    input_messages_key="query",
    history_messages_key="history"
)

for question in questions:
    response = chain_with_history.invoke(
       {
        "query": question,

       },
       config={"session_id": session}
    )
    print(f"User: {question}")
    print(f"AI: {response.content}\n")