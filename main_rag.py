from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
) 

embeddings = OpenAIEmbeddings()

files = [
  'docs/GTB_gold_Nov23.pdf',
  'docs/GTB_platinum_Nov23.pdf',
  'docs/GTB_standard_Nov23.pdf'
]

documents = sum([PyPDFLoader(file).load() for file in files], [])

peaces = RecursiveCharacterTextSplitter(
  chunk_size=1000, 
  chunk_overlap=0
  ).split_documents(documents)

recovered_data = FAISS.from_documents(peaces, embeddings).as_retriever(search_kwargs={"k": 2})

safe_consult_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer exclusively based on the provided context"),
    ("human", "{query}\n\nContext:\n{context}")
])

chain = safe_consult_prompt | model | StrOutputParser()

def answer_query(query: str):
  parts = recovered_data.invoke(query)
  context = "\n".join([part.page_content for part in parts])
  return chain.invoke({"query": query, "context": context})

print(answer_query("What is the GTB Gold Card?"))