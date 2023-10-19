import os

# Define your source and destination directories
imgs = "imgs"
captions = "captions"

# Ensure the destination directory exists; if not, create it
if not os.path.exists(captions):
    os.makedirs(captions)

# Iterate through all files in imgs
for filename in os.listdir(imgs):
    file_path = os.path.join(imgs, filename)
    
    # Check if it's a file and not a directory
    if os.path.isfile(file_path):
        # Extract the file name without its extension
        name_without_extension = os.path.splitext(filename)[0]
        
        caption = f"a 2d texture of food, {name_without_extension}"
        # Create a new file in captions with the file's name and write the name into it
        new_file_path = os.path.join(captions, f"{name_without_extension}.txt")
        with open(new_file_path, 'w') as new_file:
            new_file.write(caption)

print(f"Processed all files from {imgs} to {captions}")