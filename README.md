# Current Projects:
- GPT Architecture Language Model

## GPT Architecture Language Model
### How to Use
Get script documentation by running the following command
`python lm_model.py -h`
### Training Configurations
#### Low Performance / Test Config
1. `cd gpt`
2. `python lm_model.py -t tiktoken -m o200k_base -l <model path> -s <model path> -d <training data path> 8 8 4 1e-3 5000 4 3 0.1`
#### Current Chimera Config
1. `cd gpt`
2. `python lm_model.py -t tiktoken -m o200k_base -l <model path> -s <model path> -d <training data path> 384 64 256 3e-4 5000 6 6 0.2`
