import pandas as pd
import re
import string

def clean_code(code_sample):
    # Remove single-line comments
    code_sample = re.sub(r'#.*', '', code_sample)

    # Remove any lines that are blank or contain only spaces
    code_sample = re.sub(r'^\s*\n', '', code_sample, flags=re.MULTILINE)
    
    return code_sample

def load_data(human_code_path, GPT_code_path):
    # Load the datasets
    human_data = pd.read_csv(human_code_path, sep=',', quotechar='"', on_bad_lines='skip')
    ai_data = pd.read_csv(GPT_code_path, sep=',', quotechar='"', on_bad_lines='skip')

    # Apply cleaning to code samples in each dataset
    human_data['code'] = human_data['code'].apply(clean_code)
    ai_data['code'] = ai_data['code'].apply(clean_code)

    # Add labels to differentiate data sources
    human_data['label'] = 'human'
    ai_data['label'] = 'ai'

    # Combine the datasets
    combined_data = pd.concat([human_data, ai_data], ignore_index=True)

    # Save combined data for future use
    combined_data.to_csv('data/combined_data.csv', index=False)
    print("Combined data saved to 'data/combined_data.csv'.")

    return combined_data