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