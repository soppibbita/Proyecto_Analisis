import pandas as pd
import string

def normalize_text(text):
    # Convert to string in case of non-string values
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove spaces
    text = text.replace(' ', '')
    return text

def compare_column_values(df1, df2, column_name):
    # Normalize all values in the specified column for both dataframes
    df1_values = df1[column_name].apply(normalize_text)
    df2_values = df2[column_name].apply(normalize_text)
    
    # Convert to sets for comparison
    set1 = set(df1_values)
    set2 = set(df2_values)
    
    # Find coincidences
    coincidences = set1.intersection(set2)
    num_coincidences = len(coincidences)
    
    # Calculate percentages
    total_unique_df1 = len(set1)
    total_unique_df2 = len(set2)
    
    percentage_df1 = (num_coincidences / total_unique_df1) * 100
    percentage_df2 = (num_coincidences / total_unique_df2) * 100
    
    # Print results
    print(f"\nResults for column: {column_name}")
    print(f"Number of matching unique values found: {num_coincidences}")
    print(f"Total unique values in first dataframe: {total_unique_df1}")
    print(f"Total unique values in second dataframe: {total_unique_df2}")
    print(f"Percentage of matches in first dataframe: {percentage_df1:.2f}%")
    print(f"Percentage of matches in second dataframe: {percentage_df2:.2f}%")

# Example usage:
# Assuming 'data' and 'df2' are your dataframes and 'title' is your column name
# compare_column_values(data, df2, 'title') 