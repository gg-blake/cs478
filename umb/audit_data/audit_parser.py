"""
Name : audit_parser.py
Description : Tool to parse UMass Boston student degree audits
Author : Blake Moody
Date : 11-9-2024
"""
from io import BytesIO
from typing import Any, Dict, List, TypedDict
from pypdf import PdfReader
import regex as re
import json
import sys
import json

section_pattern = r"(NO|OK|\+\-)\s+(((\s+[A-Z0-9\:\&\/\']+)+)(\s+\*+(\s+(\s+([A-Z0-9\:\&\/\']+))+)?)?)"
subsection_pattern = r"(\s{3,}|\n\s*)(\-|\+|ip\s*(\-|\+))(\s+R\s)?(\s+[0-9]\))?\s*(([A-Z0-9(\/)\:\']([A-Za-z0-9\/\:\'\-]+\s)*)([A-Za-z0-9\/\:\-\']+))(\s{3,}|\n\s*)"
entry_pattern = r"(([a-zA-Z0-9]{4})\s+([a-zA-z]+\s?[0-9]+([a-zA-Z]?)+)\s+([0-9]+\.[0-9]+)\s+([A-Za-z\-\+\/]+)\s+(\>S|\>X|\>\-|RP|\>D|\>R)?\s+((([\w\&\/]+\s)+)?\w+(\n|[^.])?))"
reqs_pattern = r"((Needs\:)\s+([1-9]+)\s*.+(\n\s+)?)?(Select from\:)(((\s+[A-Z]+\s*)?(\,)?([0-9]+[A-Z]*)(\([A-Z0-9\s]+\))?(\,)?)+)"
req_option_pattern = r"([A-Z]+)?\s*((([0-9]+)([A-Z])?)(\s*TO\s*([0-9]+))?)"

course_catalog = {}
with open("../course_data/data.json", "r") as f:
    course_catalog = json.load(f)

print(len(course_catalog.keys()))

class Response:
    def __init__(self, status: bool, message: str):
        self.status = status
        self.message = message

    def __dict__(self):
        return {
            "status": self.status,
            "message": self.message
        }
    
    def __str__(self):
        print(self.__dict__())

def is_course_available(key):
    if key not in course_catalog.keys():
        return Response(False, "not found")
    
    if "sessions" not in course_catalog[key].keys():
        return Response(False, "no sessions available")
    
    session_available = False
    for course_session in course_catalog[key]["sessions"]:
        is_full = int(course_session["enrolled"]) >= int(course_session["capacity"])
        is_closed = course_session["status"].lower() != "open"
        if not is_full and not is_closed:
            session_available = True

    if not session_available:
        return Response(False, "all sessions are closed or unavailable")

    return Response(True, "session is available")
    
def extract_as_txt_from_request_file(file):
    pdf_bytes = BytesIO(file.read())
    
    reader = PdfReader(pdf_bytes)
    number_of_pages = len(reader.pages)

    extracted_text = ""
    for i in range(number_of_pages):
        page = reader.pages[i]
        text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
        text = re.sub(re.compile("\s+Page\s[0-9]+\sof\s[0-9]+(\s+)?"), "\n", text)
        extracted_text += text
    
    return extracted_text

def extract_as_txt(filename):
    reader = PdfReader(f"pdf/{filename}.pdf")
    number_of_pages = len(reader.pages)

    with open(f"txt/{filename}.txt", "w") as f:
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
            text = re.sub(re.compile("\s+Page\s[0-9]+\sof\s[0-9]+(\s+)?"), "\n", text)
            f.write(text)
        
        f.close()

def extract_section(pattern, text):
    s = list(re.finditer(pattern, text))
    matches = [x.start() for x in s]
    matches_shifted = matches[1:] + [len(text)]
    zipped = [text[m0:m1] for m0, m1 in zip(matches, matches_shifted)]
    
    return zip(s, zipped)

def add_entry(entry_data, group_list, section_key, entry_title: str):
    data = {
        "term": group_list[1],
        "course": group_list[2],
        "credits": group_list[4],
        "grade": group_list[5],
        "title": " ".join(str(group_list[7]).split()),
        "type": " ".join(str(section_key).split()),
        "subtype": " ".join(str(entry_title).split())
    }
    entry_data.append(data)

def add_all_entries(text: str, entry_data, section_key: str, entry_title: str) -> None:
    global entry_pattern
    entries = extract_section(entry_pattern, text)
    for entry_obj, _ in entries:
        group_list = list(entry_obj.groups())
        add_entry(entry_data, group_list, section_key, entry_title)

def add_req(key: str, section_key: str, entry_title: str, req_data):
    if key in course_catalog.keys():
        data = {
            "course": key,
            "type": " ".join(str(section_key).split()),
            "subtype": " ".join(str(entry_title).split()),
            "availability": is_course_available(key).__dict__(),
            **(course_catalog[key])
        }
    else:
        data = {
            "course": key,
            "type": " ".join(str(section_key).split()),
            "subtype": " ".join(str(entry_title).split()),
            "availability": is_course_available(key).__dict__()
        }
    req_data.append(data)

def add_all_reqs(text: str, section_key: str, entry_title: str, req_data: List[Any]) -> None:
    reqs = extract_section(reqs_pattern, text)
    course_code = ""
    for _, reqs_str in reqs:
        req_options = extract_section(req_option_pattern, reqs_str)
        for req_option_obj, req_option_str in list(req_options)[1:]:
            program_prefix = req_option_obj.group(1)
            program_suffix = req_option_obj.group(4)
            program_range = req_option_obj.group(7)
            min = int(program_suffix)
            max = min
            if program_prefix is not None:
                course_code = program_prefix

            if program_range is not None:
                max = int(program_range)

            for i in range(min, max):
                key = course_code + str(i) if i != min else course_code + req_option_obj.group(3)
                if key in course_catalog.keys():
                    add_req(key, section_key, entry_title, req_data)
            
            key = course_code + req_option_obj.group(3)
            add_req(key, section_key, entry_title, req_data)

class AuditParserOutput(TypedDict):
    entry_data: Any
    req_data: Any

def parse_audit_text(text: str) -> AuditParserOutput:
    entry_data = []
    req_data = []
    
    sections = extract_section(section_pattern, text)
    validation_length = len([_ for _ in extract_section(entry_pattern, text)])

    for section_obj, section_str in sections:
        subsections = list(extract_section(subsection_pattern, section_str))
        section_key = section_obj.groups()[2]

        if len(subsections) == 0:
            entry_title = section_key
            add_all_entries(section_str, entry_data, section_key, entry_title)
            add_all_reqs(section_str, section_key, entry_title, req_data)
            continue
        
        for subsection_obj, subsection_str in subsections:
            entry_title = subsection_obj.groups()[5]
            add_all_entries(subsection_str, entry_data, section_key, entry_title)
            add_all_reqs(subsection_str, section_key, entry_title, req_data)

    if len(entry_data) != validation_length:
        print("Warning: Validation failed for audit text parsing.")

    return {
        "entry_data": entry_data,
        "req_data": req_data
    }

if __name__ == "__main__":
    audit_name = sys.argv[1]
    extract_as_txt(audit_name)
    text = ""
    with open(f"txt/{audit_name}.txt", "r") as f:
        text = f.read()
    f.close()

    parse_result = parse_audit_text(text)

        
    with open(f"json/{audit_name}-past.json", "w+") as outfile: 
        json.dump(parse_result["entry_data"], outfile, indent=4)

    with open(f"json/{audit_name}.json", "w+") as outfile:
        json.dump(parse_result["req_data"], outfile, indent=4)

    print("Saved as", f"json/{audit_name}.json")
