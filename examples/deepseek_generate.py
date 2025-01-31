if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser

llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-...",
)

output_parser = StrOutputParser()
chain = llm | output_parser
response = chain.invoke("Sing a ballad of LangChain.")
print(response)