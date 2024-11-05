from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model="mistral",
    temperature=0,
)

db = SQLDatabase.from_uri("duckdb:///data/budget.db")


def get_sql_chain(llm, db, table_info, top_k=10):
    template = f"""Given an input question, first create a syntactically
    correct SQL query to run in {db.dialect}, then look at the results of the
    query and return the answer to the input question. You can order the
    results to return the most informative data in the database.
    
    Unless otherwise specified, do not return more than {{top_k}} rows.

    Never query for all columns from a table. You must query only the
    columns that are needed to answer the question. Wrap each column name
    in double quotes (") to denote them as delimited identifiers.

    Pay attention to use only the column names present in the tables
    below. Be careful to not query for columns that do not exist. Also, pay
    attention to which column is in which table. Query only the columns you
    need to answer the question.
    
    Please carefully think before you answer.

    Here is the schema for the database:
    {{table_info}}

    Additional info: {{input}}

    Return only the SQL query such that your response could be copied
    verbatim into the SQL terminal.
    """

    prompt = PromptTemplate.from_template(template)

    sql_chain = create_sql_query_chain(llm, db, prompt)

    return sql_chain

def natural_language_chain(question, llm, db):
    table_info = db.get_table_info()
    sql_chain = get_sql_chain(llm, db, table_info=table_info)

    template = f"""
        You are a data scientist at a compony. Based on the table schema provided
        below, the SQL query and the SQL response, provide an answer that is as
        accurate as possible. Please provide a natural language response that can
        be understood by the user. Note that the currency is in Norwegian crowns.
        Please think carefully about your answer.

        SQL Query: {{query}}
        User question: {{question}}
        SQL Response: {{response}}
        """

    prompt = PromptTemplate.from_template(template)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            response=itemgetter("query") | QuerySQLDataBaseTool(db=db)
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({"question": question})

    print(response)

    return response

_ = natural_language_chain("How many transactions are there?", llm, db)
