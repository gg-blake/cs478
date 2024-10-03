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

# Function to recursively find HTML files and process them
def process_html_files(root_directory):
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.html'):  # Check if the file is an HTML file
                file_path = os.path.join(dirpath, filename)
                
                # Read the HTML content
                with open(file_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()
                
                # Extract the text from HTML
                clean_text = extract_text_from_html(html_content)
                
                # Create a new file path for the output text file
                new_file_path = os.path.splitext(file_path)[0] + '.txt'
                
                # Write the cleaned text to the new file
                with open(new_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(clean_text)
                
                print(f'Processed {file_path} -> {new_file_path}')

# Set the root directory where the HTML files are located
root_directory = './path_to_your_directory'  # Change this to your directory path

# Run the process
process_html_files(root_directory)
