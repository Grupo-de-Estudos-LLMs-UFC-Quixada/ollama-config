import duckdb
import re
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
import pandas as pd

llm = ChatOllama(
    model="mistral",
    temperature=0,
)

db = SQLDatabase.from_uri("duckdb:///data/budget.db")

def langchain_sql_query(question, llm, db):

    chain = create_sql_query_chain(llm, db)

    response = chain.invoke({"question": question})
    print(re.search("(SELECT.*);", response.replace("\n", " ")).group(1))
    print(db.run(re.search("(SELECT.*);", response.replace("\n", " ")).group(1)))

    return response

_ = langchain_sql_query("How much did I on average spend at bunnpris?", llm, db)
