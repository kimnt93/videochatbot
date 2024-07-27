from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the log message format
)