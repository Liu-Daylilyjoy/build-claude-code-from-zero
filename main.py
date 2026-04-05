import asyncio
from typing import Any
from unittest import result

import click

from agent.agent import Agent
from agent.event import AgentEventType
from client.llm_client import LLMClient
from ui.ui import TUI, get_console

console = get_console()

class CLI:
    def __init__(self):
        self.agent: Agent | None = None
        self.tui = TUI(console)
    
    async def run_single(self, message: str):
        async with Agent() as agent:
            self.agent = agent
            await self._process_message(message)

    def _get_tool_kind(self, tool_name: str) -> str | None:
        tool_kind = None
        tool = self.agent.tool_registry.get(tool_name)
        if not tool:
            tool_kind = None

        tool_kind = tool.kind.value

        return tool_kind            
            
    async def _process_message(self, message: str) -> str | None:
        if not self.agent:
            return None
        
        assistant_stream = False
        final_response: str | None = None
        
        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                if not assistant_stream:
                    self.tui.begin_assistant()
                    assistant_stream = True
                
                self.tui.stream_assistant_delta(event.data.get("content"))
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                if assistant_stream:
                    self.tui.end_assistant()
                    assistant_stream = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown error.")
                console.print(f"\n[error]Error: {error}[/error]")
            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_start(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("arguments", {}),
                )
            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_complete(
                    event.data.get("call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("success", False),
                    event.data.get("output", ""),
                    event.data.get("metadata"),
                    event.data.get("truncated", False),
                )
        
        return final_response

@click.command()
@click.argument("prompt", required=False)
def main(prompt: str):
    cli = CLI()
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            exit(1)
    
main()