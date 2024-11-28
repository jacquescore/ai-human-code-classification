import pandas as pd


def load_data():
    print("Checking if files exist...")
    human_data = pd.read_csv('data/human_code.csv')
    print(human_data.head())
    ai_data = pd.read_csv('data/ChatGPT_pycode.csv')
    print(ai_data.head())