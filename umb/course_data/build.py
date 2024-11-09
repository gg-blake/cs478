from extract_urls import extract_urls
from urllib.request import urlopen
from urllib.parse import quote
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

semester_urls = [
    'https://courses.umb.edu/course_catalog/subjects/2025%20Spring',
]

catalog_data = {

}
for url in semester_urls:
    program_urls = [quote(i).replace("%3A", ":") for i in extract_urls(url) if "course_catalog/courses" in i]
    for x in tqdm(program_urls):
        course_urls = [quote(i).replace("%3A", ":") for i in extract_urls(x) if "course_info" in i]
        for y in tqdm(course_urls):
            soup = BeautifulSoup(urlopen(y))
            data = []
            table = soup.find_all(name='table')[-1]
            code = soup.find_all(name="h3")[-1]
            try:
                program_code = "".join(code.text.split(": ")[-1].split())
            except:
                print(code.text)
                exit()
            content = soup.find_all(name="div", attrs={"id":"body-content"})[-1]
            title = content.find_all(name="h2", attrs={"class": "page-title"})[-1].text
            course_descriptors = {s.text.split(":")[0]: s.next_sibling.next_sibling for s in content.find_all(name="p")[-1].find_all(name="strong")}
            rows = table.find_all(name="tr", attrs={"class":"class-info-rows"})
            extra_rows = table.find_all(name="tr", attrs={"class":"extra-info"})
            sessions = [{field.get('data-label'): "".join("".join(field.text.split("\t")).split("\n")) for field in row.find_all(name="td")} for row in rows]
            for i, row in enumerate(extra_rows):
                for field in row.find_all(name="span", attrs={"class":"class-div-header"}):
                    sessions[i][field.text] = "".join("".join(field.next_sibling.next_sibling.text.split("\t")).split("\n"))
            
            catalog_data[program_code] = {
                "id": program_code,
                "title": title,
                "course_descriptors": course_descriptors,
                "sessions": sessions
            }

with open("data.json", "w") as outfile: 
    json.dump(catalog_data, outfile)