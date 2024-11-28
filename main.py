from src.preprocess_data import load_data

print("Script starting...")

def main():
    print("Starting main function...")
    # Run the preprocessing function
    load_data()
    
if __name__ == '__main__':
    print("Running as main module...")
    main()