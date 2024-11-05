import duckdb
import re
from langchain_ollama import ChatOllama
import pandas as pd

llm = ChatOllama(
    model="mistral",
    temperature=0,
)

conn = duckdb.connect("data/budget.db")
cursor = conn.cursor()

schema_dict = {}
for tbl in cursor.execute("SHOW TABLES").fetchall():
    schema_dict[tbl[0]] = cursor.execute(f"DESCRIBE {tbl[0]};").fetchall()
schema_dict

def simple_prompt_engineering(question, schema, llm, conn):

    direct_prompt = f"""
    Given an input question, create a syntactically correct SQL query to
    run in DuckDB. The database is defined by the following schema:

    {schema}

    Only use the following tables:
    {conn.execute("SHOW TABLES").fetchall()}

    Don't use more columns than strictly necessary. Be careful to not
    query for columns that do not exist. Also, pay attention to which
    column is in which table. Please think carefully before you answer.

    Return only a SQL query and nothing else.

    Question: """

    response = llm.invoke(direct_prompt + question)

    print(response.content)
    result = str(conn.execute(response.content).fetchall()[0][0])
    print(f"Answer: {result}")

simple_prompt_engineering(
    "How many items did I buy at bunnpris?", schema_dict, llm, conn
)

conn.close()
