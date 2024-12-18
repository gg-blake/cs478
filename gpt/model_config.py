# Model config
EMBEDDING_SIZE=1024
BATCH_SIZE=8
BLOCK_SIZE=512
STEPS=500000
LEARNING_RATE=1e-5
HEAD_COUNT=16
LAYER_COUNT=24
DROPOUT=0.2

# Tokenizer Config
TOKENIZER_NAME="tiktoken"
TOKENIZER_MODEL="o200k_base"

# Data Config
TRAIN_DATA_PATH="data/openwebtext/test.hf"