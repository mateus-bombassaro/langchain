from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)


prompt__beach_consultant = ChatPromptTemplate.from_messages([
    ("system", "Your name is Mr. Beach. You are a travel guide specializing in Beaches"),
    ("human", "{query}")
])

prompt__mountain_consultant = ChatPromptTemplate.from_messages([
    ("system", "Your name is Mr. Mountain. You are a travel guide specializing in Mountains"),
    ("human", "{query}")
])

beach_chain = prompt__beach_consultant | model | StrOutputParser()
mountain_chain = prompt__mountain_consultant | model | StrOutputParser()

class Route(TypedDict):
  destination: Literal["Beach", "Mountain"]

prompt_router = ChatPromptTemplate.from_messages([
    ("system", "Answer only with Beach or Mountain"),
    ("human", "{query}")
])

router = prompt_router | model.with_structured_output(Route) 

class State(TypedDict):
  query: str
  route: Route
  answer: str

async def router_node(state: State, config=RunnableConfig):
  return {"destination": (await router.ainvoke({"query": state["query"]}, config))}

async def beach_node(state: State, config=RunnableConfig):
  return {"answer": (await beach_chain.ainvoke({"query": state["query"]}, config))}

async def mountain_node(state: State, config=RunnableConfig):
  return {"answer": (await mountain_chain.ainvoke({"query": state["query"]}, config))}

def choose_node(state: State):
  return "Beach" if state["route"]["destination"] == "Beach" else "Mountain"

graph = StateGraph(State)

graph.add_node("router", router_node)
graph.add_node("beach", beach_node)
graph.add_node("mountain", mountain_node)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", choose_node)

graph.add_edge("beach", END)
graph.add_edge("mountain", END)

app = graph.compile()

async def main():
  answer = await app.ainvoke({"query": "I want to visit a place in Brazil famous for it's beaches and culture."})

  print(answer["answer"])

asyncio.run(main())