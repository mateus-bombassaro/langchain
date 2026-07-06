from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel 
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destination(BaseModel):  # Describes the output format for the recommended destination
    city: str = Field("The recommended city to visit")
    reason: str = Field("Reason why it is interesting to visit that city") 

class Restaurants(BaseModel):  # Describes the output format for the recommended restaurant
    city: str = Field("The recommended city to visit")
    restaurants: str = Field("Recommended restaurants in the city")

destination_parser = JsonOutputParser(pydantic_object=Destination)
restaurants_parser = JsonOutputParser(pydantic_object=Restaurants)

city_prompt = PromptTemplate(
    template="""
    Suggest a city based on my interest in {interest}.
    {output_format}
    """,
    input_variables=["interest"],
    partial_variables={"output_format": destination_parser.get_format_instructions()}
)

restaurants_prompt = PromptTemplate(
    template="""
    Suggest popular restaurants among locals in {city}
    {output_format}
    """,
    partial_variables={"output_format": restaurants_parser.get_format_instructions()}
)

cultural_prompt = PromptTemplate(
    template="Suggest cultural activities and places in {city}"
)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=api_key
)

chain_1 = city_prompt | model | destination_parser
chain_2 = restaurants_prompt | model | restaurants_parser
chain_3 = cultural_prompt | model | StrOutputParser()

chain = chain_1 | chain_2 | chain_3

response = chain.invoke(
    {
        "interest": "beaches"
    }
)
print(response)