from tokenizers import Tokenizer

# Load the tokenizer from the local JSON file
tokenizer = Tokenizer.from_file("/home/coder/whisper-trtllm-rs/models/assets/tokenizer.json")

# Get the token ID for the Chinese comma
token_ids = tokenizer.encode('A，').ids
print(f"Token ID for '，': {token_ids}")

token_ids = tokenizer.encode('A。').ids
print(f"Token ID for '。': {token_ids}")

token_ids = tokenizer.encode('A！').ids
print(f"Token ID for '！': {token_ids}")

token_ids = tokenizer.encode('A？').ids
print(f"Token ID for '？': {token_ids}")

token_ids = tokenizer.encode('A，。！？').ids
print(f"Token ID for '？': {token_ids}")