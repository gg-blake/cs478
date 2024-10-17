from gpt.tokenizer import Tokenizer
import os
from tqdm import tqdm
import json

if __name__ == "__main__":
    # Compile the training data
    files = [file for file in os.listdir("site_data")]
    data = ""
    loader = tqdm(total=len(files), desc="Compiling Training Data: ")
    index = 0
    while index < len(files):
        with open(f"site_data/{files[index]}", "r") as f:
            data += f.read()
        index += 1
        loader.update()
    loader.close()
    
    # Initialize the tokenizer
    tokenizer = Tokenizer()

    # Train the tokenizer with max number of vocab tokens
    tokenizer.train(data, 50000)

    # Save the tokenizer instance data
    vocab_size = len(tokenizer._vocab) // 1000
    index = 1
    file_name = f"tokenizer_models/umb{vocab_size}k-{index}"
    while True:
        if not os.path.exists(file_name+".model"):
            break
        index += 1
        file_name = f"tokenizer_models/umb{vocab_size}k-{index}"
    tokenizer.save(file_name)
    print(f"Tokenizer instance data saved to {file_name}.model")

    # Encde the umb text data and save it to a json file, while recording some compression stats
    ratios = []
    total_encoded_length = 0
    total_encoded_length_untrained = 0
    json_data = {}
    loader = tqdm(total=len(files), desc="Encoding File: ")
    tokenizer_untrained = Tokenizer()
    for file in files:
        with open(f"site_data/{file}") as f:
            tokenizer_training_data = f.read()
        
        # Encode the text with an untrained and trained model
        text_a = tokenizer_untrained.encode(tokenizer_training_data)
        text_b = tokenizer.encode(tokenizer_training_data)
        json_data[file] = text_b
        
        # Update the stats
        length_diff = len(text_a) / len(text_b)
        ratios.append(length_diff)
        total_encoded_length += len(text_b)
        total_encoded_length_untrained += len(text_a)

        # Update the progress bar
        loader.update()
        
    # Save the encoded text data
    index = 1
    json_file_name = f"encoded_data/{file_name.split('/')[-1]}-{index}.json"
    while True:
        if not os.path.exists(json_file_name):
            break

        index += 1
        json_file_name = f"encoded_data/{file_name.split('/')[-1]}-{index}.json"
    json.dump(json_data, open(json_file_name, "w"))
    print(f"Encoded data saved to {json_file_name}")
    loader.close()

    # Print the compression stats
    print(f"Average Ratio: {sum(ratios)/len(ratios)}x")
    print(f"Total Compression Ratio: {total_encoded_length_untrained/total_encoded_length:.2f}x {total_encoded_length_untrained} -> {total_encoded_length}")
    
    
    
