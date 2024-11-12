"""
Name : audit_parser.py
Description : Tool to parse UMass Boston student degree audits
Author : Blake Moody
Date : 11-9-2024
"""
from pypdf import PdfReader
import regex as re
import json
import sys

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

def extract_section(text, pattern):
    s = list(re.finditer(pattern, text))
    matches = [x.start() for x in s]
    matches_shifted = matches[1:] + [len(text)]
    zipped = [text[m0:m1] for m0, m1 in zip(matches, matches_shifted)]
    
    return zip(s, zipped)

def add_entry(group_list):
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

if __name__ == "__main__":
    audit_name = sys.argv[1]
    extract_as_txt(audit_name)
    text = ""
    with open(f"txt/{audit_name}.txt", "r") as f:
        text = f.read()
    f.close()

    entry_data = []
    section_pattern = r"(NO|OK|\+\-)\s+(((\s+[A-Z0-9\:\&\/\']+)+)(\s+\*+(\s+(\s+([A-Z0-9\:\&\/\']+))+)?)?)"
    subsection_pattern = r"(\s{3,}|\n\s*)(\-|\+|ip\s*(\-|\+))(\s+R\s)?(\s+[0-9]\))?\s*(([A-Z0-9(\/)\:\']([A-Za-z0-9\/\:\'\-]+\s)*)([A-Za-z0-9\/\:\-\']+))(\s{3,}|\n\s*)"
    entry_pattern = r"(([a-zA-Z0-9]{4})\s+([a-zA-z]+\s?[0-9]+([a-zA-Z]?)+)\s+([0-9]+\.[0-9]+)\s+([A-Za-z\-\+\/]+)\s+(\>S|\>X|\>\-|RP|\>D|\>R)?\s+((([\w\&\/]+\s)+)?\w+(\n|[^.])?))"
    sections = extract_section(text, section_pattern)
    validation_length = len([_ for _ in extract_section(text, entry_pattern)])

    for section_obj, section_str in sections:
        subsections = extract_section(section_str, subsection_pattern)
        section_key = section_obj.groups()[2]
        has_subsection = False
        
        for subsection_obj, subsection_str in subsections:
            has_subsection = True
            entry_title = subsection_obj.groups()[5]
            
            entries = extract_section(subsection_str, entry_pattern)
            for entry_obj, entry_str in entries:
                group_list = list(entry_obj.groups())
                add_entry(group_list)
            
        if has_subsection:
            continue

        entries = extract_section(section_str, entry_pattern)
        entry_title = section_key
        for entry_obj, entry_str in entries:
            group_list = list(entry_obj.groups())
            add_entry(group_list)
        
    with open(f"json/{audit_name}.json", "w+") as outfile: 
        json.dump(entry_data, outfile, indent=4)

    if len(entry_data) != validation_length:
        print(f"Warning: Failed to parse {validation_length - len(entry_data)} entries")
        print("\tTotal entries:", len(entry_data), "Expected entries:", validation_length, "\n")
    else:
        print("Parsing complete! ", end="")

    print("Saved as", f"json/{audit_name}.json")
    

# (NO|OK|\+\-)\s+(((\s+[A-Z0-9\:\&\/\']+)+)(\s+\*+(\s+(\s+([A-Z0-9\:\&\/\']+))+)?)?)
# ((NO|OK|\+\-)\s+(((\s+[A-Z0-9\:\&\/\']+)+)(\s+\*+(\s+(\s+([A-Z0-9\:\&\/\']+))+)?)?)|(\n\s|\s\s|\n)(ip\s)?(\-|\+)\s{2,}([0-9]+\))?\s?([A-Za-z\/\-\:0-9]+\s(\-\s)?)+)\s(^(?:(?!((NO|OK|\+\-)\s+([A-Za-z0-9]+( [A-Za-z0-9]+)+)\s+\*+|(\n\s|\s\s|\n)(ip\s)?(\-|\+)\s{2,}([0-9]+\))?\s?([A-Za-z\/\-\:0-9]+\s(\-\s)?)+)).)*$\n){0,}
# ((NO|OK|\+\-)\s+([A-Za-z0-9]+( [A-Za-z0-9]+)+)\s+\*+|(\n\s|\s\s|\n)(ip\s)?(\-|\+)\s{2,}([0-9]+\))?\s?([A-Za-z\/\-\:0-9]+\s(\-\s)?)+)\s(^(?:(?!((NO|OK|\+\-)\s+([A-Za-z0-9]+( [A-Za-z0-9]+)+)\s+\*+|(\n\s|\s\s|\n)(ip\s)?(\-|\+)\s{2,}([0-9]+\))?\s?([A-Za-z\/\-\:0-9]+\s(\-\s)?)+)).)*$\n){0,}
# (\s{3,}|\n\s*)(\-|\+|ip\s*(\-|\+))(\s+R\s)?(\s+[0-9]\))?\s*([A-Za-z0-9]+)(\s[A-Za-z0-9]+)*(\s{3,}|\n\s*)
# (\s{3,}|\n\s*)(\-|\+|ip\s*(\-|\+))(\s+R\s)?(\s+[0-9]\))?\s*(([A-Z0-9(\/)\:\']([A-Za-z0-9\/\:\'\-]+\s)*)([A-Za-z0-9\/\:\-\']+))(\s{3,}|\n\s*)