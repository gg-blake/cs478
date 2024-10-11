import os
import itertools
import shutil
import sys

import os
from bs4 import BeautifulSoup

# Function to extract text from an HTML file
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(['script', 'style']):
        script.decompose()
    
    # Get the cleaned text
    return soup.get_text(separator=' ', strip=True)

# Walks recursively through a root directory and copies all files to a destination directory
def clean_html(source, destination):
    all_files = []
    for root, _dirs, files in itertools.islice(os.walk(source), 1, None):
        for filename in files:
            if filename.endswith('.html'):
                all_files.append(os.path.join(root, filename))
    for filename in all_files:
        with open(filename, 'r', encoding='utf-8') as file:
            html_content = file.read()
            clean_text = extract_text_from_html(html_content)
        
        # Write the cleaned text to a new file
        new_filename = os.path.splitext(filename)[0].replace("/", "-")
        with open(f"{destination}/{new_filename}.txt", 'w', encoding='utf-8') as file:
            file.write(clean_text)

        file.close()

if __name__ == '__main__':
    source = sys.argv[1]
    destination = sys.argv[2]
    clean_html(source, destination)