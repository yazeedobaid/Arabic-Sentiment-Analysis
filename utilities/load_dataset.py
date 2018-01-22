import os
import csv


def load_dataset(datasetRootDirectory):
    # Select your file delimiter
    ext = '.txt'

    # Create an empty dict
    file_dict = {}

    # data-set examples labels
    file_label = []

    # Create empty list of all txt files in data-set directory
    txt_files = []

    # Select only files with the ext extension
    for subdir, dirs, files in os.walk(datasetRootDirectory):
        for file in files:
            txt_files.append(os.path.join(subdir, file))

    # Iterate over your txt files
    for txt_file in txt_files:
        # Open them and assign them to file_dict
        with open(os.path.join(txt_file), encoding="utf-8", errors='replace') as file_object:
            label = txt_file.split('\\')[-2]
            file_dict[txt_file] = file_object.read()
            if label == 'Negative':
                file_label.append(-1)
            else:
                file_label.append(1)

    # Iterate over your dict and print the key/val pairs.
    # for i in file_dict:
    #     print(i, file_dict[i])

    return file_label, file_dict



# Function to convert a csv file to a list of dictionaries.  Takes in one variable called "variables_file"
def csv_dict_list(variables_file):
    # Create an empty dict
    file_dict = {}

    # data-set examples labels
    file_label = []

    # Open variable-based csv, iterate over the rows and map values to a list of dictionaries containing key/value pairs
    reader = csv.DictReader(open(variables_file, 'rt', encoding="utf-8"))
    counter = 0
    for line in reader:
        if line['polarity'] == '1.0':
            # file_label.append('Positive')
            file_label.append(1)
        else:
            # file_label.append('Negative')
            file_label.append(-1)
        file_dict[str(counter)] = line['text']

        counter += 1

    return file_label, file_dict
