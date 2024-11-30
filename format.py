import re
from pathlib import PureWindowsPath
from typing import Any, Tuple
import os
from urllib.parse import urlparse

import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_table_as_list(table_text):
    lines = table_text.strip().split('\n')
    if len(lines) < 3:
        return table_text  # Not enough lines for a table

    # Check if the second line contains only dashes and pipes
    if not re.match(r'^[\s\|\-]+$', lines[1]):
        return table_text  # Not a table

    headers = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
    formatted_output = []

    for line in lines[2:]:  # Skip header and separator lines
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        if len(cells) != len(headers):
            continue  # Skip malformed lines

        item_output = []
        for i, cell in enumerate(cells):
            if i == 0:
                item_output.append(f"{headers[i]}: {cell}")
            else:
                item_output.append(f"• {headers[i]}: {cell}")
        
        formatted_output.append('\n'.join(item_output))

    return '\n\n'.join(formatted_output)

def process_text_with_tables(text):
    # Find all tables in the text
    table_pattern = r'\n(\|.+\|\n\|[-\s|]+\|\n(?:\|.+\|\n)+)'
    table_matches = list(re.finditer(table_pattern, text))
    
    if not table_matches:
        return text

    formatted_parts = []
    last_end = 0

    for match in table_matches:
        # Process text before the table
        before_table = text[last_end:match.start()]
        formatted_parts.append(before_table)

        # Process the table
        table = match.group(1)
        formatted_parts.append(format_table_as_list(table))

        last_end = match.end()

    # Process text after the last table
    after_last_table = text[last_end:]
    formatted_parts.append(after_last_table)
    return '\n\n'.join(part for part in formatted_parts if part.strip())

import traceback 

def format_for_telegram(model_output):
    try:
        # Process the entire text, including embedded tables

        try:
            formatted_model_output = process_text_with_tables(model_output)
        except Exception as e:
            print(f"Error during table formatting: {str(e)}")
            #traceback.print_exc() 
            return model_output# escape_markdown_v2(model_output)
        #return escape_markdown_v3(formatted_model_output)
        return formatted_model_output
    except Exception as e:
        print(f"Error during formatting: {str(e)}")
        traceback.print_exc() 
        # If there's an error, return the original text split into chunks, with all formatting removed
        #return split_string(re.sub(r'[*_`\-\.]', '', model_output))
        return model_output