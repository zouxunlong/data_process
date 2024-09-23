import os
import re
import pandas as pd


def extract_sentences(input_path, output_path):

    df = pd.read_excel(input_path, sheet_name=list(range(21)), header=None)

    with open(output_path, "w", encoding="utf8") as fOut:
        for sheet in df.keys():
            for i, [sentence0, sentence1] in enumerate(df[sheet].loc[:, [0, 1]].values):
                if i > 0:
                    sentence1 = re.sub(
                        '^([0-9i]{1,3}\.|[•-])([^0-9])',
                        '\\2',
                        ' '.join(sentence1.split())).replace('|||', ' ').strip()
                    sentence0 = re.sub(
                        '^([0-9i]{1,3}\.|[•-])([^0-9])',
                        '\\2',
                        ' '.join(sentence0.split())).replace('|||', ' ').strip()
                    if sentence1 and sentence0:
                        fOut.write('{} ||| {}\n'.format(sentence0, sentence1))


def main(input_path):

    if os.path.isfile(input_path):
        if input_path.endswith('.xlsx'):
            output_file = input_path.replace('.xlsx', '.txt')
            extract_sentences(str(input_path), str(output_file))

    elif os.path.isdir(input_path):
        for rootdir, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.xlsx'):
                    input_file = os.path.join(rootdir, file)
                    output_file = os.path.join(
                        rootdir, file.replace('.xlsx', '.txt'))
                    extract_sentences(str(input_file), str(output_file))
    else:
        print("invalid input_file")

if __name__=="__main__":
    main('_data_in_processing/mt_data/for_extraction/categorized/xlsx/240807-Singapore Chinese Cultural Centre E-C Data_reviewed as of Aug 7.xlsx')