import re

# Read the original file
with open('form_annotator/app.py', 'r') as f:
    content = f.read()

# Read the patch file
with open('patch.py', 'r') as f:
    patch = f.read()

# Find where the batch_test method is truncated
truncated_pattern = r'def batch_test\(self\):.*?messagebox\.showinfo\(\s*"Batch Complete",\s*f"Processed \{len\(pdf_files\)\} PDF files\. Results saved to \{output_'
match = re.search(truncated_pattern, content, re.DOTALL)

if match:
    # Replace the truncated method with the complete version from patch
    fixed_content = content[:match.start()] + patch + "\n\ndef main():"
    
    # Write the fixed file
    with open('form_annotator/app.py', 'w') as f:
        f.write(fixed_content)
    print("Successfully fixed app.py")
else:
    print("Could not find the truncated method")
