# Build Claude Code from Zero #day2
学习视频: [How Claude Code Works (By Building It)](https://youtu.be/3GjE_YAs03s?si=2fdQUATQ-I71wlPL)

> 概要:
封装Agent
自定义Agent schema
命令行参数读取以及终端输出美化

需要先执行
```plaintext 
pip install click rich
```
其中click用于读取命令行参数,在代码中体现为:
```python
@click.command()
@click.argument("prompt", required=False)
def main(prompt: str):
    cli = CLI()
    if prompt:
        ...
main()
```
命令行输入```python main.py "Hello!"```,prompt便被赋值为"Hello!"


rich用于终端美化,也就是让终端能够输出高亮文本,表格等内容,优化人机交互的体验,在代码中具体体现为:
```python
from rich.console import Console
from rich.theme import Theme
from rich.rule import Rule
from rich.text import Text

from agent.agent import Agent

AGENT_THEME = Theme(
    {
        # General
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "muted": "grey50",
        "border": "grey35",
        "highlight": "bold cyan",
        # Roles
        "user": "bright_blue bold",
        "assistant": "bright_white",
        # Tools
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        # Code / blocks
        "code": "white",
    }
)

_console: Console | None = None

def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)
        
    return _console

class TUI:
    def __init__(self, console: Console | None = None):
        self.console = console or get_console() 
        self._assistant_stream_open = False
        
    def stream_assistant_delta(self, content: str) -> None:
        self.console.print(content, end="", markup=False)
        
    def begin_assistant(self) -> None:
        self.console.print()
        self.console.print(Rule(Text("Assistant", style="assistant")))
        self._assistant_stream_open = True
        
    def end_assistant(self) -> None:
        if self._assistant_stream_open:
            self.console.print()
        self._assistant_stream_open = False
```

## 核心代码
```python
class Agent:
    def __init__(self):
        self.client = LLMClient()
        
    # 跑一次agent任务
    async def run(self, message: str) -> AsyncGenerator[AgentEvent]:
        yield AgentEvent.agent_start(message)
        final_response_text: str | None = None
        
        async for event in self._agentic_loop(): 
            yield event
            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response_text = event.data.get("content")
        
        yield AgentEvent.agent_end(final_response_text)
    
    # 具体的agent任务循环
    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        response_text = ""
        async for event in self.client.chat_completion(messages, True):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(error_message=event.error or "Unknown error occurred.")
                
        if response_text:
            yield AgentEvent.text_complete(response_text)
            
    # 以下两个定义，可以使得Agent能够在with语句中初始化
    async def __aenter__(self) -> Agent:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
            self.client = None
```

## Agent相关的Schema
```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from tkinter import NO
from typing import Any

from client.response import TokenUsage

class AgentEventType(str, Enum):
    # agent
    AGENT_START = "agent_start"
    AGENT_END = "agent_END"
    AGENT_ERROR = "AGENT_ERROR"
    
    # text
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"

@dataclass
class AgentEvent:
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def agent_start(cls, message: str) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_START,
            data={"message": message}
        )
        
    @classmethod
    def agent_end(
        cls, 
        response: str | None = None, 
        usage: TokenUsage | None = None
    ) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_END,
            data={"response": response, "usage": usage.__dict__ if usage else None}
        )
        
    @classmethod
    def agent_error(
        cls, 
        error_message: str, 
        detail: dict[str, Any] | None = None
    ) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_ERROR,
            data={"error": error_message, "detail": detail or {}}
        )
        
    @classmethod
    def text_delta(cls, content: str) -> AgentEvent:
        return cls(
            type=AgentEventType.TEXT_DELTA,
            data={"content": content}
        )
    
    @classmethod
    def text_complete(cls, content: str) -> AgentEvent:
        return cls(
            type=AgentEventType.TEXT_COMPLETE,
            data={"content": content}
        )
```