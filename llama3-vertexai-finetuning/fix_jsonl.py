import json
import os

def fix_jsonl_file(input_path, output_path):
    fixed_data = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                # Convert output_text to string if it's not already
                if not isinstance(item['output_text'], str):
                    # If it's a list or other data structure, convert to JSON string
                    item['output_text'] = json.dumps(item['output_text'])
                fixed_data.append(item)
            except json.JSONDecodeError:
                print(f"Error parsing line {i}: {line[:100]}...")
                continue
    
    # Write the fixed data
    with open(output_path, 'w') as f:
        for item in fixed_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Fixed {len(fixed_data)} entries, saved to {output_path}")

# Paths to your data files
data_dir = "/Users/vishnumukundan/Documents/Duke Courses/Spring_Sem'25/LLMS/group_porject/LLMs_Finetuning/llama3-vertexai-finetuning/data"
train_data_path = os.path.join(data_dir, "train_data.jsonl")
eval_data_path = os.path.join(data_dir, "eval_data.jsonl")
train_fixed_path = os.path.join(data_dir, "train_data_fixed.jsonl")
eval_fixed_path = os.path.join(data_dir, "eval_data_fixed.jsonl")

# Fix both files
fix_jsonl_file(train_data_path, train_fixed_path)
fix_jsonl_file(eval_data_path, eval_fixed_path)