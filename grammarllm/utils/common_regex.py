import re

# Definizione regex
regex_alfanum = re.compile(r"[a-zA-Z0-9]+")  # es. "abc123"
regex_letters = re.compile(r"[a-zA-Z]+")  # es. "Hello"
regex_number = re.compile(r"\d+")  # es. "12345"
regex_decimal = re.compile(r"\d+([.,]\d+)?")  # es. "3.14"
regex_var = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")  # es. "_varName"



# Common regex_dict
regex_dict = {
    'regex_alfanum': regex_alfanum,
    'regex_letters': regex_letters,
    'regex_number': regex_number,
    'regex_decimal': regex_decimal,
    'regex_var': regex_var,
}