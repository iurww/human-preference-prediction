import csv
import json
import sys

def parse_json_field(field):
    """Parse a JSON string field, handling various edge cases."""
    try:
        return json.loads(field)
    except json.JSONDecodeError:
        # If parsing fails, return the field as a single-element list
        return [field]

def expand_rows(input_file, output_file):
    """
    Expand CSV rows where prompt, response_a, response_b are JSON arrays.
    Each element in the arrays becomes a separate row.
    Newlines in text are kept as escaped \\n, not actual newlines.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Create CSV writer with proper quoting
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        
        # Write header
        writer.writerow(fieldnames)
        
        for row in reader:
            # Parse JSON arrays
            prompts = parse_json_field(row['prompt'])
            responses_a = parse_json_field(row['response_a'])
            responses_b = parse_json_field(row['response_b'])
            
            # Find the maximum length among the three arrays
            max_len = max(len(prompts), len(responses_a), len(responses_b))
            if not(len(prompts) == len(responses_a) == len(responses_b)):
                print(f"Warning: Row {reader.line_num} has mismatched lengths: "
                      f"prompt({len(prompts)}), response_a({len(responses_a)}), response_b({len(responses_b)})",
                      file=sys.stderr)
            
            # Create expanded rows
            for i in range(max_len):
                # Build row as list to maintain column order
                new_row_values = []
                
                for field in fieldnames:
                    if field == 'prompt':
                        value = prompts[i] if i < len(prompts) else (prompts[-1] if prompts else "")
                    elif field == 'response_a':
                        value = responses_a[i] if i < len(responses_a) else (responses_a[-1] if responses_a else "")
                    elif field == 'response_b':
                        value = responses_b[i] if i < len(responses_b) else (responses_b[-1] if responses_b else "")
                    else:
                        value = row[field]
                    
                    # Ensure newlines stay as \n (escape them if needed)
                    if isinstance(value, str):
                        # Replace actual newlines with \n if any exist
                        value = value.replace('\n', '\\n').replace('\r', '\\r')
                    
                    new_row_values.append(value)
                
                writer.writerow(new_row_values)
            
            # Print progress every 1000 rows
            if reader.line_num % 1000 == 0:
                print(f"Processed {reader.line_num} rows...", file=sys.stderr)
        
        print(f"Done! Processed {reader.line_num} rows total.", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python expand_csv.py input.csv output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    expand_rows(input_file, output_file)
    print(f"Expanded CSV saved to {output_file}")