import os
import subprocess

path = "../../new_result"

# Loop through directories in the specified path
for dirs in os.listdir(path):
    new_path = os.path.join(path, dirs)

    # Skip system files like .DS_Store
    if dirs == ".DS_Store":
        continue

    # Loop through files in each directory
    for filename in os.listdir(new_path):
        file = os.path.join(new_path, filename)

        # Process only .pptx files
        if file.endswith(".pptx"):
            file2 = file.replace(".pptx", ".mp4")

            # Skip if .mp4 already exists
            if os.path.exists(file2):
                continue

            # Log the processing of the file
            print(f"Processing {file}")

            # Call the external script to convert the pptx to mp4
            subprocess.call(['python', 'test.py', '--pptx', file])
