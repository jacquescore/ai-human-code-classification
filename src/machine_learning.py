import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import re

# Function to extract features from code
def extract_code_features(code):

    # Structural Features
    lines = code.split('\n')
    num_lines = len(lines)
    avg_line_length = np.mean([len(line) for line in lines]) if num_lines > 0 else 0
    num_function_calls = len(re.findall(r'\b\w+\s*\(.*?\)', code))

    # Complexity Features
    indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    avg_indent = np.mean(indent_levels) if indent_levels else 0
    nested_indentations = sum(1 for line in lines if line.startswith("    "))
    num_loops = len(re.findall(r'\b(for|while)\b', code))
    loop_to_line_ratio = num_loops / num_lines if num_lines > 0 else 0
    num_conditionals = len(re.findall(r'\b(if|elif|else)\b', code))

    # Token & Symbol Features
    unique_identifiers = len(set(re.findall(r'\b[a-zA-Z_]\w*\b', code)))
    num_strings = len(re.findall(r'\".*?\"|\'.*?\'', code))

    # Stylistic Features
    num_camel_case = len(re.findall(r'\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b', code))
    num_snake_case = len(re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', code))

    # Function Features
    num_imports = len(re.findall(r'^\s*(import|from)\s+\w+', code, re.MULTILINE))
    num_functions = len(re.findall(r'^\s*def\s+', code, re.MULTILINE))
    func_to_line_ratio = num_functions / num_lines if num_lines > 0 else 0

    # Recursion & Complexity
    has_recursion = int(bool(re.search(r'\bdef\b.*?\b(\w+)\b.*?\1', code, re.DOTALL)))
    nested_loops = len(re.findall(r'\b(for|while)\b.*?\b(for|while)\b', code, re.DOTALL))

    # Lexical Features
    tokens = re.findall(r'\b\w+\b', code)
    avg_token_length = np.mean([len(token) for token in tokens]) if tokens else 0
    
    return {
        "num_lines": num_lines,
        "avg_line_length": avg_line_length,
        "num_function_calls": num_function_calls,
        "avg_indent": avg_indent,
        "nested_indentations": nested_indentations,
        "num_loops": num_loops,
        "loop_to_line_ratio": loop_to_line_ratio,
        "num_conditionals": num_conditionals,
        "unique_identifiers": unique_identifiers,
        "num_strings": num_strings,
        "num_camel_case": num_camel_case,
        "num_snake_case": num_snake_case,
        "num_imports": num_imports,
        "num_functions": num_functions,
        "func_to_line_ratio": func_to_line_ratio,
        "has_recursion": has_recursion,
        "nested_loops": nested_loops,
        "avg_token_length": avg_token_length,
    }

def machine_learning(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Extract features for each code sample
    features = []
    feature_names = []
    labels = []
    for _, row in data.iterrows():
        sample_features = extract_code_features(row['code'])
        features.append(sample_features)
        if not feature_names:
            feature_names = list(sample_features.keys())
        labels.append(1 if row['label'] == 'ai' else 0)  # Encode 'ai' as 1 and 'human' as 0

    # Convert features to a DataFrame
    feature_df = pd.DataFrame(features)
    feature_df['label'] = labels

    # Scatter plot with darker shade on overlap
    for feature_name in feature_names:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            feature_df['label'],
            feature_df[feature_name],
            alpha=0.1,      
            s=8,            
            color='blue'   
        )
        plt.title(f"Scatter Plot of {feature_name} by Label")
        plt.xlabel("Label (0: Human, 1: AI)")
        plt.ylabel(feature_name)
        plt.xticks(ticks=[0, 1], labels=['Human', 'AI'])
        plt.show()

    # Prepare for machine learning
    features = feature_df[feature_names].values
    labels = feature_df['label'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train classifier (Random Forest)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = classifier.predict(X_test)
    print(f"Predictions: {y_pred}") 

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Feature importance analysis
    feature_importances = classifier.feature_importances_
    for name, importance in zip(feature_names, feature_importances):
        print(f"{name}: {importance:.4f}")




#ask it to run and time each program and to obtain the output of each program?

# add more: each code sample also compares against its own features for ratio

# Run with much more data to see which types of code/tasks (long short, etc) perform worse

# should my tasks match?

# double check num of function def

# add entropy calculation? add time complexity calculation?