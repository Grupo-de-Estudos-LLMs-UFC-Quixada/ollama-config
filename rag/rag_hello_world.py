from langchain_ollama import ChatOllama
import pandas as pd

llm = ChatOllama(
    model="mistral",
    temperature=0,
)

response = llm.invoke("What is the current date?")
print(pd.DataFrame(response, columns=["variable", "value"]))

