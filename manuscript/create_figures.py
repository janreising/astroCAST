import re
from pathlib import Path

figures_directory = Path("./figures")
output_filename = "content/11-figures.tex"


# Function to format captions
def format_caption(caption: str):
    # Formats strings like (A) to \textbf{(A)}
    caption = re.sub(r'\(([A-Z])\)', r'\\textbf{(\1)}', caption)
    caption = caption.replace("%", "\\%")
    return caption


# Start writing to the output file
with open(output_filename, 'w') as output_file:
    
    filenames = list(figures_directory.glob("*.png"))
    filenames = sorted(filenames)
    for filename in filenames:
        
        name = filename.name
        base_name = filename.stem
        caption_filename = filename.with_suffix(".txt")
        
        # load caption if exists
        if not caption_filename.exists():
            formatted_caption = "\\missing{figure caption}"
        else:
            with open(caption_filename, 'r') as caption_file:
                caption = caption_file.read()
                formatted_caption = format_caption(caption)
        
        # Write the figure inclusion code to the output file
        output_file.write(
                f"\\begin{{figure}}[h!]\n"
                f"\\begin{{center}}\n"
                f"\\includegraphics[width=\\linewidth]{{figures/{name}}}\n"  # Adjust path if necessary
                f"\\end{{center}}\n"
                f"\\caption{{{formatted_caption}}}\label{{fig:{base_name}}}\n"
                f"\\end{{figure}}\n\n"
                )

print(f"Figures and captions have been written to {output_filename}.")
