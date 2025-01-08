This is an LLM-generated external dataset for the:

The Learning Agency Lab - PII Data Detection Competition
Versions
v2: Added +1000 texts with new PII information like URLs and usernames. Also, now the dataset includes the PII information as columns. Note that not all the PII information is included on the text on purpose.
Description
It contains 3382 4434 generated texts with their corresponding annotated labels in the required competition format.

Description:

document (str): ID of the essay
full_text (string): AI generated text.
tokens (string): a list with the tokens (comes from text.split())
trailing_whitespace (list): a list with boolean values indicating whether each token is followed by whitespace.
labels (list): list with token labels in BIO format