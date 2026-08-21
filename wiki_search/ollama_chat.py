"""
Setup for local LLM use using ollama, see https://ollama.com/.
"""

import asyncio
import json
from collections.abc import AsyncIterator

from mcp import Client as MCPClient
from ollama import ChatResponse, chat


async def query_mcp_client(tool: str, arguments):

    async with MCPClient("http://localhost:8110/mcp") as client:
        response = await client.call_tool(name=tool, arguments=arguments)

    return json.dumps([cont.text for cont in response.content])


class ChatResponsePrinter:
    def __init__(self):
        self.thinking = False

    def print(self, response: ChatResponse):

        if response.message.thinking:
            self.thinking = True
            print("\x1b[32m" + response.message.thinking + "\x1b[m", end="", flush=True)

        if response.message.content:
            if self.thinking:
                self.thinking = False
                print()
            print(response.message.content, end="", flush=True)

    async def print_all(self, responses: AsyncIterator[ChatResponse]):

        async for response in responses:
            self.print(response)

        print()
        self.thinking = False


class Chat:
    def __init__(self, model: str, tools: list):

        self.model = model
        self.tools = tools
        self._messages = []
        self.response_printer = ChatResponsePrinter()

    def add_message(self, content: str, images: list[str] | None = None):

        message = dict(role="user", content=content)
        if images is not None:
            message["images"] = images

        self._messages.append(message)

    async def query(self, content: str, images: list[str] | None = None):

        self.add_message(content, images)
        await self.response_printer.print_all(self.process_messages())

    async def process_messages(self) -> AsyncIterator[ChatResponse]:

        max_repeats = 3
        i = 0
        while i < max_repeats:
            i += 1
            async for chunk in self._process_messages():
                yield chunk
            if self._messages[-1]["role"] == "tool":
                continue
            else:
                break

    async def _process_messages(self) -> AsyncIterator[ChatResponse]:

        stream = chat(
            model=self.model,
            messages=self._messages,
            tools=self.tools,
            think=True,
            stream=True,
        )

        assistant_response_think = ""
        assistant_response = ""
        tool_results = []
        tool_calls = []
        for chunk in stream:
            if chunk.message.tool_calls:
                tool_calls += chunk.message.tool_calls

                _tool_results = list(
                    await asyncio.gather(
                        *(
                            query_mcp_client(
                                call.function.name, call.function.arguments
                            )
                            for call in chunk.message.tool_calls
                        )
                    )
                )
                tool_results += [
                    (name, result)
                    for name, result in zip(
                        (c.function.name for c in chunk.message.tool_calls),
                        _tool_results,
                    )
                ]

            if chunk.message.thinking:
                assistant_response_think += chunk.message.thinking
            elif chunk.message.content:
                assistant_response += chunk.message.content

            yield chunk

        if len(assistant_response_think) > 0:
            thinking_message = dict(role="assistant", thinking=assistant_response_think)
            if len(tool_calls) > 0:
                thinking_message["tool_calls"] = tool_calls
            self._messages.append(thinking_message)

        if len(tool_results) > 0:
            self._messages.extend(
                dict(role="tool", tool_name=name, content=str(result))
                for name, result in tool_results
            )

        if len(assistant_response) > 0:
            self._messages.append(dict(role="assistant", content=assistant_response))


async def main():

    async with MCPClient("http://localhost:8110/mcp") as client:
        tools = await client.list_tools()

    # see https://docs.ollama.com/api/chat#body-tools
    tools = [
        dict(
            type="function",
            function=dict(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
            ),
        )
        for tool in tools.tools
    ]

    assistant = Chat("wiki_search_assistant:latest", tools=tools)

    while True:
        _input = input("\x1b[36mAsk wiki assistant:\x1b[m ")
        if _input.lower().startswith("/quit"):
            break
        else:
            await assistant.query(_input)


if __name__ == "__main__":
    asyncio.run(main())
