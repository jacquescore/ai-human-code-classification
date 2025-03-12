import pandas as pd
import re

def clean_code(code_sample):
    # Replace '\n' with actual new line characters
    code_sample = code_sample.replace(r'\n', '\n')

    # Remove single-line comments
    code_sample = re.sub(r'#.*', '', code_sample)

    # Remove multiline comments (triple quotes)
    code_sample = re.sub(r'"""(.*?)"""', '', code_sample, flags=re.DOTALL)
    code_sample = re.sub(r"'''(.*?)'''", '', code_sample, flags=re.DOTALL)

    # Remove any lines that are blank or contain only spaces
    code_sample = re.sub(r'^\s*\n', '', code_sample, flags=re.MULTILINE)

    return code_sample

def load_data(human_code_path, GPT_code_path):
    # Load the datasets
    human_data = pd.read_csv(human_code_path, sep=',', quotechar='"', on_bad_lines='skip')
    ai_data = pd.read_csv(GPT_code_path, sep=',', quotechar='"', on_bad_lines='skip')

    # Apply cleaning function
    human_data['code'] = human_data['code'].apply(clean_code)
    ai_data['code'] = ai_data['code'].apply(clean_code)

    # Add labels to datasets
    human_data['label'] = 'human'
    ai_data['label'] = 'ai'

    # Alternate combining the datasets (for easy side by side visual comparison)
    combined_data = pd.DataFrame()
    for human, ai in zip(human_data.iterrows(), ai_data.iterrows()):
        human_row = human[1]
        ai_row = ai[1]
        combined_data = pd.concat([combined_data, pd.DataFrame([human_row]), pd.DataFrame([ai_row])], ignore_index=True)

    # Save combined data to CSV
    tasks = human_data.task
    tasks.to_csv('data/tasks.csv', index=False)
    combined_data.to_csv('data/combined_data.csv', index=False)
    print("Combined data saved to 'data/combined_data.csv'.")

    return combined_data
