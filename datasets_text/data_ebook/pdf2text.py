import fitz
import os


dir = "/home/zxl/ssd/pdf2text/google_book.id/"
files = os.listdir(dir)
for file in files:
    if os.path.exists(dir+file.replace(".pdf", ".2.txt")):
        continue
    try:
        doc = fitz.open(dir+file)
        text = ""
        for i, page in enumerate(doc):
            text += page.get_text() + "\n\n"
        open(dir+file.replace(".pdf", ".2.txt"), "w", encoding="utf8").write(text)
        print("complete {}".format(file), flush=True)
    except Exception as e:
        print(e, flush=True)
        print("error on {}".format(file), flush=True)


