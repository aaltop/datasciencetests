"""
Setup for local LLM use using ollama, see https://ollama.com/.
"""

from collections.abc import Iterator

from ollama import ChatResponse, chat


def divide(x: float, y: float) -> float:
    """
    Divide two floating point values.

    Args:
        x:
            The numerator.
        y:
            The denominator.

    Returns:
        The result of the division.
    """

    return x / y


class ChatResponsePrinter:
    def __init__(self):

        self._thinking = False

    def print(self, response: ChatResponse):

        if response.message.thinking:
            if not self._thinking:
                self._thinking = True
                print("\x1b[32m", end="")

            print(response.message.thinking, end="", flush=True)

        if response.message.content:
            if self._thinking:
                self._thinking = False
                print("\x1b[m")

            print(response.message.content, end="", flush=True)

    def print_all(self, responses: Iterator[ChatResponse]):

        for response in responses:
            self.print(response)

        print()


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

    def query(self, content: str, images: list[str] | None = None):

        self.add_message(content, images)
        self.response_printer.print_all(self.process_messages())

    def process_messages(self):

        max_repeats = 3
        i = 0
        while i < max_repeats:
            i += 1
            yield from self._process_messages()
            if self._messages[-1]["role"] == "tool":
                continue
            else:
                break

    def _process_messages(self) -> Iterator[ChatResponse]:

        stream = chat(
            model=self.model,
            messages=self._messages,
            tools=self.tools,
            think=True,
            stream=True,
        )

        assistant_response_think = ""
        assistant_response = ""
        tool_result = None
        tool_calls = None
        for chunk in stream:
            if chunk.message.tool_calls:
                tool_calls = chunk.message.tool_calls
                call = chunk.message.tool_calls[0]
                tool_result = divide(**call.function.arguments)

            if chunk.message.thinking:
                assistant_response_think += chunk.message.thinking
            elif chunk.message.content:
                assistant_response += chunk.message.content

            yield chunk

        if len(assistant_response_think) > 0:
            thinking_message = dict(role="assistant", thinking=assistant_response_think)
            if tool_calls is not None:
                thinking_message["tool_calls"] = [tool_calls[0]]
            self._messages.append(thinking_message)

        if tool_result is not None:
            self._messages.append(
                dict(
                    role="tool", tool_name=call.function.name, content=str(tool_result)
                )
            )

        if len(assistant_response) > 0:
            self._messages.append(dict(role="assistant", content=assistant_response))


def main():

    assistant = Chat("wiki_search_assistant:latest", tools=[divide])
    assistant.query(
        "What tools do you have available to you? Give the summary of each."
    )


if __name__ == "__main__":
    main()
