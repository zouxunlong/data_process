import pandas as pd
import os


def xlsx_to_txt(file_in, file_out):
    df = pd.read_excel(file_in, header=None)
    with open(file_out, 'w', encoding='utf8') as file_out:
        for i, [sentence0, sentence1] in enumerate(df.loc[0:].values):
            try:
                file_out.write("{} | {}\n".format(
                ' '.join(str(sentence0).replace("|", " ").split()), ' '.join(sentence1.replace("|", " ").split())))
            except TypeError as err:
                print(err)

def convert_dir(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.xlsx'):
                xlsx_to_txt(os.path.join(root, file),
                            os.path.join(root, file)+'.en-zh')


rootdir = '/home/xuanlong/dataclean/data/MCI/en-zh/Batch_13_extracted'
convert_dir(rootdir)
