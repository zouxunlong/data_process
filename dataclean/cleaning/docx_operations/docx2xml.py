from docx import Document

wordDoc = Document("/home/xunlong/dataclean/data/Batch8(CD8)_extracted/From Agencies/MSEDenguePreventionCampaigntermsEM.docx")
doc_body = wordDoc.element.xml

with open('demo.xml', 'w') as file:
    file.write(doc_body)
