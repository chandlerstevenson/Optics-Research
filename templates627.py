import os
import pandas as pd

def find_csv_files_with_string(directory, search_string):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and search_string in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

def read_first_row(csv_file):
    df = pd.read_csv(csv_file, nrows=1)
    return df.iloc[0]

def main(directory, search_string):
    csv_files = find_csv_files_with_string(directory, search_string)
    for csv_file in csv_files:
        first_row = read_first_row(csv_file)
        print(f"First row of {csv_file}:")
        print(first_row)
        print("\n")

if __name__ == "__main__":
    directory = "path/to/your/directory"  # Replace with your directory path
    search_string = "AAAA"  # Replace with your search string
    main(directory, search_string)



import os
import pandas as pd

def find_csv_files_with_string(directory, search_string):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and search_string in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

def read_first_column(csv_file):
    df = pd.read_csv(csv_file, usecols=[0])
    return df.iloc[:, 0]

def main(directory, search_string):
    csv_files = find_csv_files_with_string(directory, search_string)
    for csv_file in csv_files:
        first_column = read_first_column(csv_file)
        print(f"First column of {csv_file}:")
        print(first_column)
        print("\n")

if __name__ == "__main__":
    directory = "path/to/your/directory"  # Replace with your directory path
    search_string = "AAAA"  # Replace with your search string
    main(directory, search_string)



import os
import pandas as pd

def find_csv_files_with_string(directory, search_string):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and search_string in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

def read_third_column_fourth_row(csv_file):
    df = pd.read_csv(csv_file, usecols=[2])
    if len(df) >= 4:
        return df.iloc[3, 0]
    else:
        return None

def main(directory, search_string):
    csv_files = find_csv_files_with_string(directory, search_string)
    for csv_file in csv_files:
        value = read_third_column_fourth_row(csv_file)
        if value is not None:
            print(f"Value in third column, fourth row of {csv_file}: {value}")
        else:
            print(f"{csv_file} does not have enough rows.")

if __name__ == "__main__":
    directory = "path/to/your/directory"  # Replace with your directory path
    search_string = "AAAA"  # Replace with your search string
    main(directory, search_string)
