from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

messages = [
    SystemMessage(content="Você é um assistente útil que traduz do Inglês para o Português. Traduza a sequência a seguir."),
    HumanMessage(content="I love programming."),
]

output = llm.invoke(messages)
print("Saída da invocação direta do Ollama:")
print(output)
print()

parser = StrOutputParser()
chain = llm | parser
output = chain.invoke(messages)
print("Saída da chain conectando o Ollama ao parser:")
print(output)
print()

system_template = "You are a translator that only outputs the translation requested. Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
chain = prompt_template | llm | parser
output = chain.invoke({
    "language": "spanish", 
    "text": "I love programming."
})
print("Saída da chain conectando o Ollama ao parser e ao prompt:")
print(output)
print()

