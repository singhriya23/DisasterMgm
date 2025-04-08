import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def markdown_to_json(md_file_path, output_json_path, chunk_size=512, chunk_overlap=50):
    """
    Reads a Markdown (.md) file, chunks the text using LangChain's RecursiveCharacterTextSplitter,
    and converts the chunks into JSON format.

    Args:
        md_file_path (str): Path to the Markdown file.
        output_json_path (str): Path to save the output JSON file.
        chunk_size (int): The size of each chunk (default: 512).
        chunk_overlap (int): Overlapping tokens between chunks (default: 50).

    Returns:
        dict: JSON object containing the chunked Markdown text.
    """

    # ✅ Validate input file format
    if not md_file_path.endswith(".md"):
        raise ValueError("Unsupported file format. This function only processes .md (Markdown) files.")

    # ✅ Ensure the output directory exists
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create directories if they don't exist

    # ✅ Read Markdown file
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Markdown file not found: {md_file_path}")
    except Exception as e:
        raise Exception(f"Error reading Markdown file: {str(e)}")

    # ✅ Chunk the text using LangChain
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)

    # ✅ Convert to JSON format
    json_output = {"chunks": [{"id": i + 1, "content": chunk} for i, chunk in enumerate(chunks)]}

    # ✅ Save JSON output to file
    try:
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(json_output, json_file, indent=2, ensure_ascii=False)
        print(f"✅ JSON file successfully saved at: {output_json_path}")
    except Exception as e:
        raise Exception(f"Error writing JSON file: {str(e)}")

    return json_output

# Example Usage
json_output = markdown_to_json(
    "DOCLING_PDF_PLUMBER_Markdowns/10K10Q-Q3-2025-with-image-refs.md", 
    output_json_path="output-json/output-q3-2025.json",  # Ensure this directory exists or will be created
    chunk_size=512, 
    chunk_overlap=50
)

# ✅ Print a preview of the output JSON
print("\n💡 JSON Output (Preview):", json.dumps(json_output, indent=2)[:500])  # Preview first 500 characters
