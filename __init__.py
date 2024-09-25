import os

# Define the directories and output file
src_directory = "./src"  # Folder containing your .py files
main_file = "main.py"  # Path to main.py outside of src folder
output_file = "combined_code_output.py"

# Open the output file in write mode
with open(output_file, "w") as outfile:
    # First, add the files from the src directory
    for filename in os.listdir(src_directory):
        if filename.endswith(".py") and filename != "__init__.py":
            file_path = os.path.join(src_directory, filename)
            with open(file_path, "r") as infile:
                # Write the content of each Python file from src
                outfile.write(f"# --- {filename} ---\n")
                outfile.write(infile.read())
                outfile.write("\n\n")

    # Next, add the main.py file at the end (or you can move it to the top by reversing these two blocks)
    if os.path.exists(main_file):
        with open(main_file, "r") as mainfile:
            outfile.write(f"# --- main.py ---\n")
            outfile.write(mainfile.read())
            outfile.write("\n\n")
    else:
        print(f"main.py not found in the specified directory")

print(
    f"All Python files from '{src_directory}' and 'main.py' have been combined into '{output_file}'."
)
