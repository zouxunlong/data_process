import subprocess
import os


def pdf2docx_with_english_ocr_adobe(filepath):

    try:

        file_name = os.path.splitext(filepath)[0]

        returncode = subprocess.call(
            ['node', 'src/exportpdf/export-pdf-to-docx.js', filepath, '{}.docx'.format(file_name)])

        if returncode != 0:
            print('failed convert {}'.format(filepath), flush=True)

    except Exception as e:
        print(e, flush=True)


def doc2docx(filepath):

    try:
        outdir = os.path.dirname(filepath)
        returncode = subprocess.call(
            ['soffice', '--headless', '--convert-to', 'docx:MS Word 2007 XML', '--infilter=MS Word 97', '--outdir', outdir, filepath])

        if returncode != 0:
            print('failed convert {}'.format(filepath), flush=True)

    except Exception as e:
        print(e, flush=True)


def rtf2docx(filepath):

    try:
        outdir = os.path.dirname(filepath)
        returncode = subprocess.call(
            ['soffice', '--headless', '--convert-to', 'docx:MS Word 2007 XML', '--outdir', outdir, filepath])

        if returncode != 0:
            print('failed convert {}'.format(filepath), flush=True)

    except Exception as e:
        print(e, flush=True)


def convert_files_in_dir(rootdir):

    file_converted = 0

    for root, dirs, files in os.walk(rootdir):
        files.sort()
        for file in files:
            if file.endswith('.pdf'):
                pdf2docx_with_english_ocr_adobe(os.path.join(root, file))
                file_converted += 1
            if file.endswith('.doc'):
                doc2docx(os.path.join(root, file))
                file_converted += 1

    print("Done. {} file converted".format(file_converted))

