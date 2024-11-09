from urllib.request import urlopen
from bs4 import BeautifulSoup

def extract_urls(url):
    page = urlopen(url).read()
    soup = BeautifulSoup(page)
    soup.prettify()
    return [anchor['href'] for anchor in soup.findAll('a', href=True)]

if __name__ == "__main__":
    print(extract_urls('https://courses.umb.edu/course_catalog/subjects/2025%20Spring'))