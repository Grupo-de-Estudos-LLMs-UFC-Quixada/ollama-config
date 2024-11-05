import duckdb
import re
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
import pandas as pd

llm = ChatOllama(
    model="mistral",
    temperature=0,
)

db = SQLDatabase.from_uri("duckdb:///data/budget.db")

def with_a_little_langchain(question, verbose=False):
    direct_prompt = f"""
    Given an input question, create a syntactically correct SQL query to
    run in DuckDB. The database is defined by the following schema:

    {db.get_table_info()}

    Only use the following tables:
    {db.get_usable_table_names()}

    Pay attention to use only the column names you can see in the tables
    below. Be careful to not query for columns that do not exist. Also, pay
    attention to which column is in which table. Please carefully think
    before you answer.

    Question:
    """

    response = llm.invoke(direct_prompt + question)

    if verbose:
        print(response)

    print(re.search("(SELECT.*);", response.content.replace("\n", " ")).group(1))
    print(
        db.run(re.search("(SELECT.*);", response.content.replace("\n", " ")).group(1))
    )

with_a_little_langchain("How many items did I buy at bunnpris in 2023?", verbose=True)
