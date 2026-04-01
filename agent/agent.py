from __future__ import annotations
from typing import AsyncGenerator

from agent.event import AgentEvent, AgentEventType
from client import response
from client.llm_client import LLMClient
from client.response import StreamEventType


class Agent:
    def __init__(self):
        self.client = LLMClient()
        
    async def run(self, message: str) -> AsyncGenerator[AgentEvent]:
        yield AgentEvent.agent_start(message)
        final_response_text: str | None = None
        
        async for event in self._agentic_loop(): 
            yield event
            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response_text = event.data.get("content")
        
        yield AgentEvent.agent_end(final_response_text)
    
    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        messages = [{"role": "user", "content": "What's the matter with you?!"}]
        
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
            
    # for with statement
    async def __aenter__(self) -> Agent:
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()
            self.client = None