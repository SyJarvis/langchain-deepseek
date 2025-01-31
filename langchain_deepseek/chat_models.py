import inspect
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Mapping,
    Tuple,
    Iterator,
    Type
)

from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream
)

from langchain_core.messages import (
    BaseMessage,
    ChatMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    BaseMessageChunk,
    AIMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    FunctionMessageChunk,
    ToolMessageChunk,
    ChatMessageChunk
)

from langchain_core.outputs import (
    ChatResult,
    ChatGeneration, ChatGenerationChunk
)

from langchain_core.callbacks import (
    CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
)

from pydantic import Field, BaseModel
import json
from .api import RestAPI


class ChatDeepSeek(BaseChatModel):
    # TODO: Replace all TODOs in docstring. See example docstring:
    # https://github.com/langchain-ai/langchain/blob/7ff05357bac6eaedf5058a2af88f23a1817d40fe/libs/partners/openai/langchain_openai/chat_models/base.py#L1120
    """DeepSeek chat model integration.

    The default implementation echoes the first `parrot_buffer_length` characters of the input.

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``langchain_deepseek`` and set environment variable ``DEEPSEEK_API_KEY``.

        .. code-block:: bash

            pip install -U langchain_deepseek
            export DEEPSEEK_API_KEY="your-api-key"

    """  # noqa: E501
    """deepseek client"""
    client: Any = Field(default=None, exclude=True)
    """model name use"""
    model: str = Field("deepseek-chat")
    api_key: Optional[str] = Field(None)
    base_url: Optional[str] = None
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1
    request_id: Optional[str] = None
    max_tokens: Optional[int] = 2048
    streaming: Optional[bool] = False
    n: Optional[int] = 1
    response_format: Dict[str, str] = {"type": "text"}
    frequency_penalty: Optional[int] = 0
    presence_penalty: Optional[int] = 0
    tools: Any = None
    tool_choice: Optional[str] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"deepseek_api_key": "DEEPSEEK_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Return"""
        return ["langchain", "chat_models", "DeepSeek"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        return attributes

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-deepseek"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model,
        }

    @classmethod
    def filter_model_kwargs(cls):
        """
        """
        return [
            "model",
            "frequency_penalty",
            "max_tokens",
            "presence_penalty",
            "response_format",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "logprobs",
            "top_logprobs",
            "request_id"
        ]
    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            stream: Optional[bool] = None,
            **kwargs: Any
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        message_dict, params = self._create_message_dicts(messages, stop)
        response = self.completion_with_retry(
            message_dict=message_dict,
            run_manager=run_manager,
            params=params
        )
        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dict, params = self._create_message_dicts(messages, stop)
        params["stream"] = True if "stream" in params else False
        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry_stream(
            message_dict=message_dict, run_manager=run_manager, params=params
        ):
            if "choices" not in chunk or len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            delta = choice["delta"]
            delta.update({"id": chunk["id"]})
            message_chunk = _convert_delta_to_message_chunk(
                delta, default_class=default_chunk_class
            )
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info["created"] = chunk["created"]
                generation_info["model"] = chunk["model"]
            if usage := chunk.get("usage"):
                generation_info["usage"] = usage
            cg_chunk = ChatGenerationChunk(
                message=message_chunk, generation_info=generation_info
            )
            if run_manager is not None:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    def completion_with_retry(
            self, message_dict=None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs
    ):
        params = kwargs["params"]
        params.update({"messages": message_dict})
        params.update({"stream": False})
        try:
            self.client = RestAPI(base_url=self.base_url, api_key=self.api_key)
            reply = self.client.action_post(request_path=f"chat/completions", **params)
        except Exception as e:
            raise e
        return reply

    def completion_with_retry_stream(
            self, message_dict=None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs
    ):
        params = kwargs["params"]
        params.update({"messages": message_dict})
        params.update({"stream": True})
        try:
            self.client = RestAPI(base_url=self.base_url, api_key=self.api_key)
        except Exception as e:
            raise e
        response = self.client.action_post(request_path=f"chat/completions", **params)
        for line in response.iter_lines():
            if line:
                line_utf8 = line.decode("utf-8")
                if line_utf8.startswith("event:"):
                    continue
                elif line_utf8.startswith("data:"):
                    text = line_utf8[5:].strip()
                    if text == '[DONE]':
                        break
                    else:
                        dict_text = json.loads(text)
                        if dict_text and isinstance(dict_text, dict):
                            yield dict_text

    def _create_chat_result(self, response):
        generations = []
        id = response.get("id")
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message_dict = res["message"]
            message = _convert_dict_to_message(message_dict)
            generation_info = dict(finish_reason=res.get("finish_reason"))
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "id": id,
            "created": response.get("created"),
            "token_usage": token_usage,
            "model_name": self.model,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
            self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self.get_model_kwargs()
        params.update({"stream": False})
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params.update({"stop": stop})
        message_dicts = [_convert_message_to_dict(message) for message in messages]
        # 传递prompt
        params.update({"messages": message_dicts})
        return message_dicts, params

    def get_model_kwargs(self):
        attrs = {}
        for cls in inspect.getmro(self.__class__):
            attrs.update(vars(cls))
        attrs.update((vars(self)))
        return {
            attr: value for attr, value in attrs.items() if attr in self.__class__.filter_model_kwargs() and value is not None
        }


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        if tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    elif role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))  # type: ignore[arg-type]
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),  # type: ignore[arg-type]
            additional_kwargs=additional_kwargs,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role)  # type: ignore[arg-type]


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]