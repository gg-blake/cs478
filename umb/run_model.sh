#! /bin/bash
while true
do
    python model.py website_clean.txt tokenizer_models/umb100k-1.model 100000 website_full_text.pth
done
