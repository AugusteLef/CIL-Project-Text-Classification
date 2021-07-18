# This script takes paths to all three datasets (the original one and two additional ones)
# and combines them into one large dataset.

import argparse
import pandas as pd

# custom file imports
import utils

def combine(path1: str, path2: str, path3: str) -> pd.DataFrame:
    """Combine 3 differents sets of tweets into one. 
    Use to creat a bigger training set based on 2 additional datasets. 

    Args:
        path1 (str): path to the 1st dataset to combine
        path2 (str): path to the 2nd dataset to combine
        path3 (str): path to the 3rd dataset to combine

    Returns:
        pd.DataFrame: a DataFrame of the 3 datasets concatenated together
    """
    data1 = utils.load_raw_data(path1)
    data2 = utils.load_raw_data(path2)
    data3 = utils.load_raw_data(path3)
    
    combined_data_temp = pd.concat([data1, data2])
    combined_data =  pd.concat([combined_data_temp, data3])
    
    return combined_data

def save_combined(data: pd.DataFrame, output_path: str):
    """Save the combined dataset in the correct format and with given output file name

    Args:
        data (pd.DataFrame): [Dataset to save]
        output_path (str): [output file name]
    """
    data.to_csv("Data/" + output_path, header=None, index=None, sep=',',quotechar=" ", mode='a', line_terminator=' ')
    
def main(args):
    dataFrame = combine(args.input_path1, args.input_path2, args.input_path3)
    save_combined(dataFrame, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='concatenate/combine 3 datasets composed of tweets into one huge dataset')
    parser.add_argument('input_path1', type=str, help='path to raw data', action='store')
    parser.add_argument('input_path2', type=str, help='path to raw data', action='store')
    parser.add_argument('input_path3', type=str, help='path to raw data', action='store')
    parser.add_argument('output_path', type=str, help='name of the output file', action='store')
    args = parser.parse_args()
    main(args)

