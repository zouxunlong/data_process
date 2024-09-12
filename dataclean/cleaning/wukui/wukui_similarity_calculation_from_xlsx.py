import pandas as pd
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("LaBSE")

df = pd.read_excel('english_to_chinese.xlsx', header=None)

dfList = []

for i, [sentence0, sentence1, sentence2, sentence3] in enumerate(df.loc[0:].values):

    # Compute embeddings
    embedding0 = model.encode(sentence0, convert_to_tensor=True)
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    embedding3 = model.encode(sentence3, convert_to_tensor=True)

    # Compute cosine-similarities for each sentence with each other sentence
    cosine_score01 = util.cos_sim(embedding0, embedding1)
    cosine_score02 = util.cos_sim(embedding0, embedding2)
    cosine_score03 = util.cos_sim(embedding0, embedding3)

    dfList.append([sentence0, sentence1, sentence2, sentence3, 
    cosine_score01[0][0].item(), cosine_score02[0][0].item(), cosine_score03[0][0].item()])

df_new =  pd.DataFrame(dfList)
df_new.to_excel('output.xlsx', index=False, header=False)
