import pandas as pd

def is_valid_csv(file_path, expected_columns):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Check if the expected columns are present and in order
        if df.columns.tolist() != expected_columns:
            return False, "CSV does not have the expected columns or column order."

        # Check if 'ans' values are single integers within the range 0 to 17
        for i, value in enumerate(df['ans']):
            if not (isinstance(value, int) and 0 <= value <= 17):
                return False, f"in a line {i}, Value '{value}' in 'ans' column is out of range or not an integer."

        return True, "CSV format is valid."

    except Exception as e:
        return False, f"Error reading CSV file: {e}"

# Example usage
file_path = './output.csv'
expected_columns = ['ImageID', 'ans']

is_valid, message = is_valid_csv(file_path, expected_columns)
print(message)