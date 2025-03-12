from src.preprocess_data import load_data
from src.machine_learning import machine_learning

def main():
    # Run data preprocessing function
    data = load_data("data/small_human_code.csv", "data/small_GPT_code.csv")
    # Run machine learning function
    machine_learning("data/combined_data.csv")

if __name__ == '__main__':
    main()