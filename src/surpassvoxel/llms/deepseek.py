from typing import Any, Literal
from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from surpassvoxel.utils.file_handles import get_config


class DeepSeekChatConfig(BaseModel, frozen=True):
    name: str = Field(default="deepseek")
    url: str = Field(default="https://api.deepseek.com")
    model: Literal["deepseek-flash", "deepseek-v4-pro"] = Field(default="deepseek-flash")
    effort: Literal["none", "low", "high", "max"] = Field(default="high")
    json_object: bool = Field(default=False)


class DeepSeekChatModel(ChatOpenAI):
    use_responses_api: bool | None = False

    @staticmethod
    def _reasoning_content(raw_message: Any) -> str | None:
        if isinstance(raw_message, dict):
            return raw_message.get("reasoning_content")
        return getattr(raw_message, "reasoning_content", None)

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(warnings=False)
        )
        choices = response_dict.get("choices") or []
        for generation, choice in zip(result.generations, choices):
            reasoning_content = self._reasoning_content(choice.get("message") or {})
            if reasoning_content:
                generation.message.additional_kwargs["reasoning_content"] = reasoning_content
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type[BaseMessageChunk],
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        delta = choices[0].get("delta") if choices else None
        reasoning_content = delta.get("reasoning_content") if delta else None
        if reasoning_content:
            generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_content
        return generation_chunk

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for message, body in zip(messages, payload.get("messages") or []):
            if not isinstance(message, AIMessage):
                continue
            reasoning_content = message.additional_kwargs.get("reasoning_content")
            if reasoning_content:
                body["reasoning_content"] = reasoning_content
        return payload


def get_deepseek_chat_model(config: DeepSeekChatConfig = DeepSeekChatConfig()) -> DeepSeekChatModel:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "base_url": config.url,
        "api_key": get_config("llm_keys")[config.name],
        "extra_body": {"thinking": {"type": "disabled" if config.effort == "none" else "enabled"}},
    }
    if config.effort != "none":
        kwargs["reasoning_effort"] = config.effort
    if config.json_object:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return DeepSeekChatModel(**kwargs)
