# Build Claude Code from Zero #day1
学习视频: [How Claude Code Works (By Building It)](https://youtu.be/3GjE_YAs03s?si=2fdQUATQ-I71wlPL)

> 概要:
封装LLM client
自定义消息schema
掌握流式/非流式chat completion

只需要```pip install openai```即可运行

## 核心代码
```python
class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
    
    # 使用AsyncOpenAI而不是OpenAI，便于异步调用
    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key="<your-api-key-here>",
                base_url="<base_url, eg.: https://openrouter.ai/api/v1>"
            )
        
        return self._client

    async def close(self) -> None:
        ...
                
    # 使用AsyncGenerator配合yield进行逐块输出，用户体验更好
    async def chat_completion(self, messages: list[dict[str, Any]], stream: bool) -> AsyncGenerator[StreamEvent, None]:
        # 设置llm参数，stream为bool值，表示是否使用流式传输
        client = self.get_client()
        kwargs = {
            "model": "<model-name>",
            "messages": messages,
            "stream": stream
        }
        
        if stream:
            async for event in self._stream_response(client, kwargs):
                yield event
        else: 
            event = await self._non_stream_response(client, kwargs)
            yield event
            
        return
    
    # 此函数的写法适用于OpenAI格式的输出，如果报错，需要观察一下自己的模型的response结果
    async def _stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(
            **kwargs
        )
        
        finish_reason = None
        usage = None
        
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
                    type=EventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content)
                )
    
        yield StreamEvent(
            EventType.MESSAGE_COMPLETE,
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
        
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens
            )
        
        return StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage
        )

```
## 自定义消息Schema
```python
@dataclass
class TextDelta:
    content: str
    
    def __str__(self):
        return self.content
    
@dataclass
class EventType(str, Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE = "message_complete"
    ERROR = "error"

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    
    def __add__(self, other: TokenUsage):
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens
        )

@dataclass
class StreamEvent:
    type: EventType
    text_delta: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
```