import os


def split_file_by_lang(file, output_file1, output_file2):
    with open(file, encoding='utf8') as f_in, open(output_file1, 'w', encoding='utf8') as f_out1, open(output_file2, 'w', encoding='utf8') as f_out2:
        for i, line in enumerate(f_in):
            sentences = line.split('|')
            if len(sentences) != 2:
                return
            f_out1.write(sentences[0].strip()+'\n')
            f_out2.write(sentences[1].strip()+'\n')


def combine_files_in_dir(rootdir):

    file_splited = 0

    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.en-ta') or file.endswith('.EN-TA'):
                file = os.path.join(root, file)
                split_file_by_lang(file, file+'.en', file+'.ta')
                file_splited += 1
            if file.endswith('.en-ms') or file.endswith('.EN-MS'):
                file = os.path.join(root, file)
                split_file_by_lang(file, file+'.en', file+'.ms')
                file_splited += 1
            if file.endswith('.en-zh') or file.endswith('.EN-ZH'):
                file = os.path.join(root, file)
                split_file_by_lang(file, file+'.en', file+'.zh')
                file_splited += 1
            if file.endswith('.ta-en') or file.endswith('.TA-EN'):
                file = os.path.join(root, file)
                split_file_by_lang(file, file+'.ta', file+'.en')
                file_splited += 1
            if file.endswith('.ms-en') or file.endswith('.MS-EN'):
                file = os.path.join(root, file)
                split_file_by_lang(file, file+'.ms', file+'.en')
                file_splited += 1
            if file.endswith('.zh-en') or file.endswith('.ZH-EN'):
                file = os.path.join(root, file)
                split_file_by_lang(file, file+'.zh', file+'.en')
                file_splited += 1
    print("Done. {} file splited".format(file_splited))


rootdir = '/home/xuanlong/dataclean/data/MCI_combined'
combine_files_in_dir(rootdir)
