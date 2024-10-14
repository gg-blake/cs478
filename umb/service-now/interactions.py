from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import regex as re

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
#chrome_options.add_argument("--no-sandbox")  # Needed for some systems
#chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

# Path to your ChromeDriver
webdriver_service = Service('/usr/bin/chromedriver')

# Set up Chrome options to use the existing user data
chrome_options.add_argument("user-data-dir=/mnt/c/Users/Blake/AppData/Local/Google/Chrome/User Data")  # Change to your path
chrome_options.add_argument("profile-directory=Blake")  # Adjust if using a different profile

# Initialize the WebDriver
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(['script', 'style']):
        script.decompose()
    
    # Get the cleaned text
    return soup.get_text(separator=' ', strip=True)

try:
    # Navigate to the page
    driver.get('https://umassboston.service-now.com/incident.do?sys_id=08d646e047115210a11e59f2846d43be&sysparm_record_target=incident&sysparm_record_row=1&sysparm_record_rows=208797&sysparm_record_list=ORDERBYDESCopened_at')  # Replace with your target URL

    # Wait for a specific element that indicates the page is fully loaded
    # Adjust the selector to match an element that is added by JavaScript
    # Switch to the specific frame or iframe
    #driver.switch_to.frame('Main Content')  # You can also use an element here
    #incident.description
    # Wait for an element within the frame to ensure it's fully loaded
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.TAG_NAME, 'macroponent-f51912f4c700201072b211d4d8c26010'))  # Change this selector
    )
    # Wait for the page to fully load (you might need to adjust this)
    time.sleep(15)

    # Locate the host element that contains the shadow DOM
    host_element = driver.find_element(By.TAG_NAME, 'macroponent-f51912f4c700201072b211d4d8c26010')  # Adjust as needed
    pattern = "[A-Za-z0-9]+ [A-Za-z0-9]+•[0-9]{4}-[0-9]{2}-[0-9]{2}\s[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,3})?\s[A-Za-z]+"
    compiled_pattern = re.compile(pattern)
    
    
    # Access the shadow root using JavaScript and return all elements within it
    iframe = driver.execute_script("""
        const shadowRoot = arguments[0].shadowRoot;
        return shadowRoot.getElementById("gsft_main");
    """, host_element)

    driver.switch_to.frame(iframe)
    description = driver.find_element(By.ID, "sn_form_inline_stream_entries")
    output = re.split(compiled_pattern, description.text) # Split the text at points where the regex pattern matches
    output = [data for data in output if data is not None] # Remove the None values fromthe list
    print(output)
    filtered_text = "\n<New Event>".join(output)
    print(filtered_text)

    '''time.sleep(10)
    shadow_root = driver.find_element(By.TAG_NAME, 'macroponent-f51912f4c700201072b211d4d8c26010').shadow_root
    
    # Get all elements within the shadow root
    all_elements = driver.execute_script('return Array.from(arguments[0].querySelectorAll("*"))', shadow_root)'''

    # After the window is closed, you can scrape the page source
    updated_html = driver.page_source

    with open(f"output.txt", 'w', encoding='utf-8') as file:
        file.write(updated_html)

    print("Source code saved to output.txt")

    

finally:
    driver.quit()  # Close the browser