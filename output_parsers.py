from langchain_core.output_parsers import BaseOutputParser
from typing import List, Any, Optional, Dict, Tuple
import re

from format import format_for_telegram


class TelegramOutputParser(BaseOutputParser):
    def parse(self, output: Any) -> List[str]:
        if isinstance(output, dict):
            text = output.get('answer', '') or output.get('result', '') or str(output)
        elif isinstance(output, str):
            text = output
        else:
            text = str(output)
        return format_for_telegram(text)

    def get_format_instructions(self) -> str:
        return "Format the output for Telegram using markdown."

    @property
    def _type(self) -> str:
        return "telegram_output"
    

def parse_llama_response(llama_output: str) -> str:
    """
    Parses the output from a Llama 3.1 model and extracts only the assistant's response.
    If the expected special tokens are not found, returns the original output.

    Args:
        llama_output (str): The raw output string from the Llama 3.1 model, including special tokens.

    Returns:
        str: The concatenated assistant responses as a single string, or the original output if patterns are absent.
    """
    # Define regex patterns for identifying headers and messages
    header_pattern = re.compile(r'<\|start_header_id\|>(.*?)<\|end_header_id\|>')
    message_split_pattern = re.compile(r'<\|start_header_id\|>.*?<\|end_header_id\|>')

    # Check if the special header patterns exist in the llama_output
    if not header_pattern.search(llama_output):
        # If patterns are not found, return the original llama_output
        return llama_output.strip()

    # Find all headers and their corresponding positions
    headers = [
        (match.group(1).strip(), match.start(), match.end()) 
        for match in header_pattern.finditer(llama_output)
    ]

    assistant_responses = []

    for i, (role, start, end) in enumerate(headers):
        if role.lower() == 'assistant':
            # Determine the start of the message
            message_start = end

            # Determine the end of the message
            if i + 1 < len(headers):
                message_end = headers[i + 1][1]
            else:
                message_end = len(llama_output)

            # Extract the message content
            message = llama_output[message_start:message_end]

            # Remove any special tokens within the message
            message = re.sub(r'<\|[^|]+\|>', '', message).strip()

            # Append the cleaned message to the list
            assistant_responses.append(message)

    # Concatenate all assistant responses into a single string
    return "\n".join(assistant_responses)


class LlamaOutputParser(TelegramOutputParser):
    def parse(self, output: Any) -> List[str]:
        if isinstance(output, dict):
            text = output.get('answer', '') or output.get('result', '') or str(output)
        elif isinstance(output, str):
            text = output
        else:
            text = str(output)
        return super().parse(parse_llama_response(text))
   
    def get_format_instructions(self) -> str:
        return "Format the output from Llama model to dict of ['input', 'answer', context']."
   
    @property
    def _type(self) -> str:
        return "llama_output"
