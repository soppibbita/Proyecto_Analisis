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

def merge_unique_values(df1, df2):
    # Create copies to avoid modifying original dataframes
    df1_copy = df1.copy()
    df2_copy = df2.copy()
    
    # Rename 'author' to 'authors' in first dataframe
    if 'author' in df1_copy.columns:
        df1_copy = df1_copy.rename(columns={'author': 'authors'})
    
    # First, find rows in df1 that have unique titles (using normalized comparison)
    df1_copy['normalized_title'] = df1_copy['title'].apply(normalize_text)
    df2_copy['normalized_title'] = df2_copy['title'].apply(normalize_text)
    
    # Get set of normalized titles from both dataframes
    titles_df1 = set(df1_copy['normalized_title'])
    titles_df2 = set(df2_copy['normalized_title'])
    
    # Find titles that are unique to df1
    unique_titles = titles_df1 - titles_df2
    
    if len(unique_titles) > 0:
        # Get the original rows from df1 that have unique titles
        unique_mask = df1_copy['normalized_title'].isin(unique_titles)
        unique_rows = df1_copy[unique_mask].copy()
        
        # Remove the temporary normalized_title column
        unique_rows = unique_rows.drop('normalized_title', axis=1)
        
        # Add these rows to df2
        df2_copy = pd.concat([df2_copy.drop('normalized_title', axis=1), unique_rows], ignore_index=True)
        
        print(f"Added {len(unique_rows)} new unique rows from the first dataset")
        print(f"Original size of second dataset: {len(df2_copy) - len(unique_rows)}")
        print(f"New size of second dataset: {len(df2_copy)}")
    else:
        print("No unique entries found in the first dataset")
        df2_copy = df2_copy.drop('normalized_title', axis=1)
    
    return df2_copy

# Example usage:
# Assuming 'data' is your first dataframe and 'df2' is your second dataframe
# df2_updated = merge_unique_values(data, df2)
# df2_updated.to_csv('updated_dataset.csv', index=False)  # Save the updated dataframe 