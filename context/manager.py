from dataclasses import dataclass, field
from typing import Any

from prompts.system import get_system_prompt
from utils.text import count_tokens

@dataclass
class MessageItem:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    token_usage: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        result:dict[str, Any] = {"role": self.role}
        
        if self.content:
            result["content"] = self.content
            
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
            
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
            
        return result

class ContextManager:
    def __init__(self):
        self._system_prompt = get_system_prompt()
        self._messages: list[MessageItem] = []
        self._model_name: str = "qwen/qwen3.6-plus-preview:free"
        
    def add_user_message(self, content: str) -> None:
        self._messages.append(
            MessageItem(
                role="user",
                content=content,
                token_usage=count_tokens(
                    content,
                    self._model_name
                )
            )
        )

    def add_assistant_message(self, content: str | None) -> None:
        self._messages.append(
            MessageItem(
                role="assistant",
                content=content or "",
                token_usage=count_tokens(
                    content or "",
                    self._model_name
                )
            )
        )
        
    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(
            MessageItem(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
                token_usage=count_tokens(content, self._model_name)
            )
        )
    
    def get_messages(self) -> list[dict[str, Any]]:
        messages = []
        
        if self._system_prompt:
            messages.append({
                "role": "system",
                "content": self._system_prompt
            })
        
        for message in self._messages:
            messages.append(message.to_dict())
            
        return messages