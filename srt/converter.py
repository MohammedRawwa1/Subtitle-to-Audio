import re

def srt_to_txt(srt_file, txt_file):
    # Regular expression to match timestamps in the SRT format
    timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}'

    with open(srt_file, 'r', encoding='utf-8') as srt, open(txt_file, 'w', encoding='utf-8') as txt:
        content = srt.read()
        # Remove the timestamp lines and empty lines
        clean_content = re.sub(timestamp_pattern, '', content)
        clean_content = re.sub(r'\n\s*\n', '\n', clean_content).strip()  # Remove extra empty lines
        
        # Write the cleaned content to the txt file
        txt.write(clean_content)

    print(f"Conversion complete! The text has been saved to {txt_file}")

# Example usage
srt_to_txt('19- FAQ.srt', '19- FAQ.txt')
