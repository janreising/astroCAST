import re
from pathlib import Path


def fix_language_codes(bbl_file_path: Path, output_file_path: Path = None):
    """Replace 'en' and 'eng' with 'english' in .bbl files.

    Args:
      bbl_file_path: Path to the original .bbl file.
      output_file_path: Path to save the modified .bbl file. If None, overwrite the original.

    Raises:
      FileNotFoundError: If the original .bbl file does not exist.
    """
    
    with open(bbl_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace 'en' and 'eng' with 'english', ensuring we don't match parts of words
    content = re.sub(r'language = \{en\}', r'language = {english}', content)
    content = re.sub(r'language = \{eng\}', r'language = {english}', content)
    
    if output_file_path is None:
        output_file_path = bbl_file_path
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content)


file_path = Path("astroCAST.bib")
assert file_path.exists(), f"{file_path} does not exist"

fix_language_codes(file_path, file_path)
