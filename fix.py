import re

# Path to your .srt file
file_path = r"C:\Users\Mohammad Rawwas\Downloads\Compressed\subtitle-to-audio-main\srt\١٩-أسئلة شائعة.srt"

# Read the content of the file
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Regular expression to find timestamps with excessive precision
pattern = r"(\d{2}:\d{2}:\d{2},\d{3})\.\d+"  # Match timestamps with excessive precision
fixed_content = re.sub(pattern, r"\1", content)  # Replace with standard format

# Check if the pattern was found and fixed
if fixed_content != content:
    print("Timestamps fixed successfully!")
else:
    print("No excessive precision found. No changes were made.")

# Write the fixed content back to the file
with open(file_path, "w", encoding="utf-8") as file:
    file.write(fixed_content)

print("File saved successfully.")