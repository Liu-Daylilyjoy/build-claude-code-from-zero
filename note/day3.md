# Build Claude Code from Zero #day3
学习视频: [How Claude Code Works (By Building It)](https://youtu.be/3GjE_YAs03s?si=2fdQUATQ-I71wlPL)

> 概要:
工具调用

## 核心代码
### 简单的上下文管理器
```python
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

# 核心思想是将历史聊天记录保存在内存dict中,每次调用llm时,传入历史记录
# 在agent的init函数中初始化此类
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
```
### 工具注册器
```python
import logging
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolInvocation, ToolResult
from tools.builtin import get_all_builtin_tools

logger = logging.getLogger(__name__)

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")

        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> bool:
        if name in self.tools:
            del self.tools[name]
            return True
        
        return False
    
    def get(self, name: str) -> Tool | None:
        if name in self._tools:
            return self._tools[name]

        return None

    def get_tools(self) -> list[Tool]:
        tools: list[Tool] = []

        for tool in self._tools.values():
            tools.append(tool)

        return tools

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self.get_tools()]
    
    async def invoke(
        self,
        name: str,
        params: dict[str, Any],
        cwd: Path,
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            result = ToolResult.error_result(
                f"Unknown tool: {name}",
                metadata={"tool_name": name},
            )
            return result

        # 校验参数,具体方式是使用try-except语句结合pydantic
        validation_errors = tool.validate_params(params)
        if validation_errors:
            result = ToolResult.error_result(
                f"Invalid parameters: {'; '.join(validation_errors)}",
                metadata={
                    "tool_name": name,
                    "validation_errors": validation_errors,
                },
            )

            return result

        invocation = ToolInvocation(
            params=params,
            cwd=cwd,
        )

        try:
            result = await tool.execute(invocation)
        except Exception as e:
            logger.exception(f"Tool {name} raised unexpected error")
            result = ToolResult.error_result(
                f"Internal error: {str(e)}",
                metadata={
                    "tool_name", name
                }
            )

        return result

# 在agent的init函数中调用此函数
def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    for tool_class in get_all_builtin_tools(): 
        registry.register(tool_class())

    return registry
```
get_all_builtin_tools写在init函数中,可以很方便的导出同目录下所有的工具:
```python
from tools.builtin.read_file import ReadFileTool
__all__ = [
    "ReadFileTool"
]
def get_all_builtin_tools() -> list[type]:
    return [
        ReadFileTool
    ]
```

---
修改agent流程,增加上下文以及工具调用分支
```python
    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        response_text = ""
        tool_schemas = self.tool_registry.get_schemas() # 新增
        tool_calls: list[ToolCall] = [] # 新增
        async for event in self.client.chat_completion(self.context_manager.get_messages(), tools=tool_schemas if tool_schemas else None):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content)
            # 记录下所有的tool_call,响应结束后统一处理
            elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                if event.tool_call:
                    tool_calls.append(event.tool_call)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(error_message=event.error or "Unknown error occurred.")
                
        # 新增
        self.context_manager.add_assistant_message(
            response_text or None
        )       
        
        # 新增
        if response_text:
            yield AgentEvent.text_complete(response_text)
            
        tool_call_results: list[ToolResultMessage] = []

        # 新增 
        for tool_call in tool_calls:
            yield AgentEvent.tool_call_start(
                tool_call.call_id,
                tool_call.name,
                tool_call.arguments
            )
            
            result = await self.tool_registry.invoke(tool_call.name, tool_call.arguments, Path.cwd())
            
            yield AgentEvent.tool_call_complete(
                tool_call.call_id,
                tool_call.name,
                result
            )
            
            tool_call_results.append(
                ToolResultMessage(
                    tool_call_id=tool_call.call_id,
                    content=result.to_model_output(),
                    is_error=not result.success
                )
            )
        
        for tool_call_result in tool_call_results:
            self.context_manager.add_tool_result(tool_call_result.tool_call_id, tool_call_result.content)
```

---
llm是怎么操纵工具的?
在调用时将tool列表传给llm(使用openai标准格式)
```python
if tools:
    kwargs["tools"] = self._build_tools(tools)
    kwargs["tool_choice"] = "auto"
```
llm会自动决定是否tool_call(llm的返回中有这一字段)
需要修改读取函数
```python
    async def _stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(
            **kwargs
        )
        
        finish_reason = None
        usage = None
        tool_calls: dict[str, Any] = {}
        
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens
                )
                
            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
            
            if choice.finish_reason:
                finish_reason = choice.finish_reason    
                                
            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content)
                )
                
            # 如果llm决定调用函数,tool_calls将不为空，由于返回值分了多块，因此参数需要循环拼接
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index

                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tool_call_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            # function和id是一起出现的
                            tool_calls[idx]["name"] = tool_call_delta.function.name
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tool_calls[idx]["id"],
                                    name=tool_call_delta.function.name,
                                ),
                            )

                    if tool_call_delta.function.arguments:
                        tool_calls[idx][
                            "arguments"
                        ] += tool_call_delta.function.arguments

                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            tool_call_delta=ToolCallDelta(
                                call_id=tool_calls[idx]["id"],
                                name=tool_calls[idx]["name"],
                                arguments_delta=tool_call_delta.function.arguments,
                            ),
                        )

        # 此处并不是真正执行,而是tool_call参数封装完成,准备任务
        for idx, tc in tool_calls.items():
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=tc["id"],
                    name=tc["name"],
                    arguments=parse_tool_call_arguments(tc["arguments"]),
                ),
            )
    
        yield StreamEvent(
            StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage
        )
        
    
    async def _non_stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> StreamEvent:
        response = await client.chat.completions.create(
            **kwargs
        )
        
        choice = response.choices[0]
        message = choice.message
        
        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)
            
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append([
                    ToolCall(
                        tc.id,
                        tc.function.name,
                        arguments=parse_tool_call_arguments(tc.function.arguments)
                    )
                ])
        
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens
            )
        
        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage
        )
```

---
使用pydantic约束tool schema
```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic.json_schema import model_json_schema

class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"
    
@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    truncated: bool = False
    
    @classmethod
    def error_result(
        cls,
        error: str,
        output: str = "",
        **kwargs: Any
    ) -> ToolResult:
        return cls(
            success=False,
            output=output,
            error=error,
            **kwargs
        )
        
    @classmethod
    def success_result(cls, output: str, **kwargs: Any):
        return cls(
            success=True,
            output=output,
            error=None,
            **kwargs,
        )
        
    def to_model_output(self) -> str:
        if self.success:
            return self.output

        return f"Error: {self.error}\n\nOutput:\n{self.output}"
        
@dataclass
class ToolInvocation:
    params: dict[str, Any]
    cwd: Path
    
@dataclass
class ToolConfirmation:
    tool_name: str
    params: dict[str, Any]
    description: str
    
# 所有tool,都需要继承此类
class Tool(ABC):
    name: str = "base_tool"
    description: str = "Base tool"
    kind: ToolKind = ToolKind.READ
    
    def __init__(self):
        pass
    
    # @property
    # 把 schema 变成属性访问，而不是方法调用。
    # 也就是说你会写 tool.schema，而不是 tool.schema()。
    # -> dict[str, Any] | type["BaseModel"]
    # 这是类型注解，表示这个属性返回值应该是两种之一：
    # 一个字典，比如 JSON Schema 那样的结构
    # 一个 BaseModel 类本身（通常是 Pydantic 模型类）
    @property
    def schema(self) -> dict[str, Any] | type["BaseModel"]:
        raise NotImplementedError("Tool must define schema property or class attribute")
    
    @abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        pass
    
    def validate_params(self, params: dict[str, Any]) -> list[str]:
        schema = self.schema
        
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema(**params)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    errors.append(f"Parameter '{field}': {msg}")

                return errors
            except Exception as e:
                return [str(e)]

        return []
    
    def is_mutating(self, params: dict[str, Any]) -> bool:
        return self.kind in {
            ToolKind.WRITE,
            ToolKind.SHELL,
            ToolKind.NETWORK,
            ToolKind.MEMORY
        }
        
    async def get_confirmation(self, invocation: ToolInvocation) -> ToolConfirmation | None:
        if not self.is_mutating(invocation.params):
            return None
        
        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute {self.name}"
        )
        
    # openai 格式工具调用
    def to_openai_schema(self) -> dict[str, Any]:
        schema = self.schema

        # 处理“schema 是 Pydantic 模型类”的情况
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = model_json_schema(schema, mode="serialization")

            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", []),
                },
            }
            
        # 处理“schema 已经是字典 schema”的情况
        if isinstance(schema, dict):
            result = {
                "name": self.name,
                "description": self.description,
            }

            if "parameters" in schema:
                result["parameters"] = schema["parameters"]
            else:
                result["parameters"] = schema

            return result

        raise ValueError(f"Invalid schema type for tool {self.name}: {type(schema)}")
```

---
第一个工具: read_file文件阅读,让llm知道文件内容
```python
from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import is_binary_file, resolve_path
from utils.text import count_tokens, truncate_text

# 此处继承BaseModel类,并使用pydantic的参数,Field中定义该参数的行为与描述
class ReadFileParams(BaseModel):
    path: str = Field(
        ...,
        description="Path to the file to read (relative to working directory or absolute)",
    )

    offset: int = Field(
        1,
        ge=1,
        description="Line number to start reading from (1-based). Defaults to 1",
    )

    limit: int | None = Field(
        None,
        ge=1,
        description="Maximum number of lines to read. If not specified, reads entire file.",
    )
    
# 这是一个单例
class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a text file. Returns the file content with line numbers. "
        "For large files, use offset and limit to read specific portions. "
        "Cannot read binary files (images, executables, etc.)."
    )
    kind = ToolKind.READ

    schema = ReadFileParams

    MAX_FILE_SIZE = 1024 * 1024 * 10
    MAX_OUTPUT_TOKENS = 25000

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ReadFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        if not path.exists():
            return ToolResult.error_result(f"File not found: {path}")

        if not path.is_file():
            return ToolResult.error_result(f"Path is not a file: {path}")

        file_size = path.stat().st_size

        if file_size > self.MAX_FILE_SIZE:
            return ToolResult.error_result(
                f"File too large ({file_size / (1024*1024):.1f}MB). "
                f"Maximum is {self.MAX_FILE_SIZE / (1024*1024):.0f}MB."
            )

        if is_binary_file(path):
            file_size_mb = file_size / (1024 * 1024)
            size_str = (
                f"{file_size_mb:.2f}MB" if file_size_mb >= 1 else f"{file_size} bytes"
            )
            return ToolResult.error_result(
                f"Cannot read binary file: {path.name} ({size_str}) "
                f"This tool only reads text files."
            )

        try:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")

            lines = content.splitlines()
            total_lines = len(lines)

            if total_lines == 0:
                return ToolResult.success_result(
                    "File is empty.",
                    metadata={
                        "lines": 0,
                    },
                )

            start_idx = max(0, params.offset - 1)

            if params.limit is not None:
                end_idx = min(start_idx + params.limit, total_lines)
            else:
                end_idx = total_lines

            selected_lines = lines[start_idx:end_idx]
            formatted_lines = []

            for i, line in enumerate(selected_lines, start=start_idx + 1):
                formatted_lines.append(f"{i:6}|{line}")

            output = "\n".join(formatted_lines)
            token_count = count_tokens(output, "qwen/qwen3.6-plus:free")

            truncated = False
            if token_count > self.MAX_OUTPUT_TOKENS:
                output = truncate_text(
                    output,
                    self.MAX_OUTPUT_TOKENS,
                    suffix=f"\n... [truncated {total_lines} total lines]",
                )
                truncated = True

            metadata_lines = []
            if start_idx > 0 or end_idx < total_lines:
                metadata_lines.append(
                    f"Showing lines {start_idx+1}-{end_idx} of {total_lines}"
                )

            if metadata_lines:
                header = " | ".join(metadata_lines) + "\n\n"
                output = header + output

            return ToolResult.success_result(
                output=output,
                truncated=truncated,
                metadata={
                    "path": str(path),
                    "total_lines": total_lines,
                    "shown_start": start_idx + 1,
                    "shown_end": end_idx,
                },
            )
        except Exception as e:
            return ToolResult.error_result(f"Failed to read file: {e}")
```
### 工具类
```python
from pathlib import Path


def resolve_path(base: str | Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    
    return (Path(base).resolve() / path).resolve()

def display_path_rel_to_cwd(path: str, cwd: Path | None) -> str:
    try:
        p = Path(path)
    except Exception:
        return path

    if cwd:
        try:
            return str(p.relative_to(cwd))
        except ValueError:
            pass

    return str(p)

def is_binary_file(path: str | Path) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except (OSError, IOError):
        return False

```
```python
import tiktoken

def get_tokenizer(model: str):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode

def count_tokens(text: str, model: str = "gpt-4") -> int:
    tokenizer = get_tokenizer(model)

    if tokenizer:
        return len(tokenizer(text))

    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)

def truncate_text(
    text: str,
    model: str,
    max_tokens: int,
    suffix: str = "\n... [truncated]",
    preserve_lines: bool = True,
):
    current_tokens = count_tokens(text, model)
    if current_tokens <= max_tokens:
        return text

    suffix_tokens = count_tokens(suffix, model)
    target_tokens = max_tokens - suffix_tokens

    if target_tokens <= 0:
        return suffix.strip()

    if preserve_lines:
        return _truncate_by_lines(text, target_tokens, suffix, model)
    else:
        return _truncate_by_chars(text, target_tokens, suffix, model)


def _truncate_by_lines(text: str, target_tokens: int, suffix: str, model: str) -> str:
    lines = text.split("\n")
    result_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line + "\n", model)
        if current_tokens + line_tokens > target_tokens:
            break
        result_lines.append(line)
        current_tokens += line_tokens

    if not result_lines:
        # Fall back to character truncation if no complete lines fit
        return _truncate_by_chars(text, target_tokens, suffix, model)

    return "\n".join(result_lines) + suffix


def _truncate_by_chars(text: str, target_tokens: int, suffix: str, model: str) -> str:
    # Binary search for the right length
    low, high = 0, len(text)

    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens(text[:mid], model) <= target_tokens:
            low = mid
        else:
            high = mid - 1

    return text[:low] + suffix
```

---
还有一些关于UI的信息,无需过多讲解,主要流程是在main中捕获tool_call_start和tool_call_complete事件并使用rich进行美观输出