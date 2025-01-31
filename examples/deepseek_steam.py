# -*- coding: utf-8 -*-
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_deepseek import ChatDeepSeek


llm = ChatDeepSeek(model="deepseek-chat")
for chunk in llm.stream("Sing a ballad of LangChain."):
    print(chunk.content, end="|", flush=True)

