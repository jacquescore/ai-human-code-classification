from src.preprocess_data import load_data

print("Script starting...")

def main():
    print("Starting main function...")
    # Run the preprocessing function
    data = load_data("data/human_code.csv", "data/GPT_code.csv")
    print(f"Data shape: {data.shape}")  # For confirmation

if __name__ == '__main__':
    print("Running as main module...")
    main()