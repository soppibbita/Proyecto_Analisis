import pandas as pd

def check_and_drop_columns(df):
    # Print all column names
    print("Available columns in the dataframe:")
    for col in df.columns:
        print(f"- {col}")
    
    # List of columns we want to drop
    columns_to_drop = ["cdrom", "note", "publnr", "publisher", "month", "stream"]
    
    # Check which columns actually exist
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    non_existing = [col for col in columns_to_drop if col not in df.columns]
    
    # Print information about which columns exist and which don't
    print("\nColumns that will be dropped:")
    for col in existing_columns:
        print(f"- {col}")
    
    if non_existing:
        print("\nColumns not found in the dataframe:")
        for col in non_existing:
            print(f"- {col}")
    
    # Drop only the existing columns
    df_clean = df.drop(columns=existing_columns)
    
    print(f"\nOriginal number of columns: {len(df.columns)}")
    print(f"Number of columns after dropping: {len(df_clean.columns)}")
    
    return df_clean

# Example usage:
# df_clean = check_and_drop_columns(df2_updated) 