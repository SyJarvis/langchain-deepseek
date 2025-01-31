# 🦜️🔗 langchain-deepseek

## Install

* Install from source (Recommend)

```bash
cd langchain-deepseek
pip install -e .
```
* Install from PyPI
```bash
pip install langchain-deepseek
```

## Quick Start
* All the code can be found in the `examples`
* Set DeepSeek API key in environment if using DeepSeek models: `export DEEPSEEK_API_KEY="sk-...".`
*  Maybe you can try loading environment variables like this. Create a new `.env` file
```
DEEPSEEK_API_KEY="sk-..."
```
```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
```
`ChatDeepSeek` class exposes chat models from DeepSeek.

```python
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat")
llm.invoke("Sing a ballad of LangChain.")
```

