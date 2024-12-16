import config

from AIAssistantsLib.assistants import RAGAssistantGPT, RAGAssistantGemini, RAGAssistantGGUF, RAGAssistantSber, RAGAssistantLocal, RAGAssistantMistralAI, RAGAssistantYA, load_vectorstore

import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    from argparse import (
        ArgumentParser,
        ArgumentDefaultsHelpFormatter,
        BooleanOptionalAction,
    )
    from output_parsers import TelegramOutputParser, LlamaOutputParser
    vectorestore_path = 'data/vectorstore_e5'

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'mode', 
        nargs='?', 
        default='query', 
        choices = ['query'],
        help='query - query vectorestore\n'
    )
    args = vars(parser.parse_args())
    mode = args['mode']

    with open('prompts/system_prompt_markdown_3.txt', 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    if mode == 'query':
        assistants = []
        vectorstore = load_vectorstore(vectorestore_path, config.EMBEDDING_MODEL)
        assistants.append(RAGAssistantGPT(system_prompt, vectorestore_path, output_parser=TelegramOutputParser))

        query = ''

        while query != 'stop':
            print('=========================================================================')
            query = input("Enter your query: ")
            if query != 'stop':
                for assistant in assistants:
                    try:
                        reply = assistant.ask_question(query)
                    except Exception as e:
                        logging.error(f'Error: {str(e)}')
                        continue
                    print(f'{reply['answer']}')
                    print('=========================================================================')
