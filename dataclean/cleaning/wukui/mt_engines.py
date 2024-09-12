
import json
import os
import pandas as pd
import requests
import aiohttp
import asyncio
import time



def sgtt(sentences_src, src, tgt):

    sentences_tgt = []
    source=src+"_SG"
    target=tgt+"_SG"
    batch_size=10
    for i in range(0, len(sentences_src), batch_size):
        batch_sentences_src = sentences_src[i:i+batch_size]
        response = requests.post(url = 'http://10.2.56.190:5008/translator',
                                 json={"source":source, 
                                       "target":target,
                                       "query": '\n'.join(batch_sentences_src)
                                       }
                                       )
        batch_sentences_tgt=[item["translatedText"] for item in response.json()["data"]["translations"]]
        sentences_tgt.extend(batch_sentences_tgt)
    assert len(sentences_src) == len(sentences_tgt), 'length of source and target do not match'

    return sentences_tgt


def itranslate(sentence_src, src, tgt):
    response = requests.post(url='https://dev-api.itranslate.com/translation/v2/',
                             headers={
                                 "Authorization": "Bearer 24784523-65ac-4c08-abcf-1ad90d398fe1",
                                 "Content-Type": "application/json"
                             },
                             json={
                                 "source": {"dialect": src, "text": sentence_src},
                                 "target": {"dialect": tgt}
                             }
                             )
    sentence_tgt = response.json()["target"]["text"]
    return sentence_tgt


def google(sentences_src, src, tgt):
    response = requests.post(url='https://translation.googleapis.com/language/translate/v2',
                             headers={
                                 "X-goog-api-key": "AIzaSyCCNyGLGl8F55oSDtIPWiigx9RJqolQDJE",
                                 "Content-Type": "application/json; charset=utf-8"
                             },
                             json={
                                 "q": sentences_src,
                                 "source": src,
                                 "target": tgt
                             }
                             )
    sentence_tgt = [item['translatedText']
                    for item in response.json()["data"]["translations"]]
    return sentence_tgt


def bing(sentences_src, src, tgt):
    sentences_tgt=[]
    response = requests.post(url='https://api.cognitive.microsofttranslator.com/translate',
                             headers={'Ocp-Apim-Subscription-Key': "1655824396af4af4bb14d2cc5428cace",
                                      'Ocp-Apim-Subscription-Region': "southeastasia",
                                      'Content-type': 'application/json',
                                      },
                             params={'api-version': '3.0',
                                     'from': src,
                                     'to': tgt
                                     },
                             json=[{'text': sentence} for sentence in sentences_src]
                             )
    for item in response.json():
        sentences_tgt.append(item["translations"][0]["text"])
    return sentences_tgt


def main_itranslate():
    for rootdir, dirs, files in os.walk("/home/xunlong/dataclean/cleaning/wukui/files2"):
        for file in files:
            src = file[:2]
            tgt = file[3:5]
            src = 'zh-CN' if src == 'zh' else src
            tgt = 'zh-CN' if tgt == 'zh' else tgt

            df = pd.read_excel(os.path.join(rootdir, file))
            for i, sentence_src in enumerate(df.iloc[:, 0].values):
                sentence_itranslate = itranslate(sentence_src, src, tgt)
                df.loc[i, "iTranslate"] = sentence_itranslate

            df.to_excel(os.path.join(rootdir, file), index=False, header=True)

    print("finished all", flush=True)


def main_google():
    for rootdir, dirs, files in os.walk("/home/xunlong/dataclean/cleaning/wukui/files3"):
        for file in files:
            src = file[:2]
            tgt = "ms"
            df = pd.read_excel(os.path.join(rootdir, file)).fillna('')
            batch_size = 100
            for i in range(0, df.shape[0], batch_size):
                sentences_src = list(df.iloc[i:i+batch_size, 0].values)
                sentences_google = google(sentences_src, src, tgt)
                df.loc[i:i+batch_size-1, "Google Translation"] = sentences_google
            df.to_excel(os.path.join(rootdir, file), index=False, header=True)
    print("finished all", flush=True)


def main_bing():
    for rootdir, dirs, files in os.walk("/home/xunlong/dataclean/cleaning/wukui/files3"):
        for file in files:
            src = file[:2]
            tgt = file[3:5]
            src = 'zh-Hans' if src == 'zh' else src
            tgt = 'zh-Hans' if tgt == 'zh' else tgt
            df = pd.read_excel(os.path.join(rootdir, file)).fillna('')
            batch_size = 200
            for i in range(0, df.shape[0], batch_size):
                sentences_src = list(df.iloc[i:i+batch_size, 0].values)
                sentences_bing = bing(sentences_src, src, tgt)
                df.loc[i:i+batch_size-1, "Microsoft Translator"] = sentences_bing
            df.to_excel(os.path.join(rootdir, file), index=False, header=True)
    print("finished all", flush=True)


def main_bing2(file, src, tgt):
    src = 'zh-Hans' if src == 'zh' else src
    tgt = 'zh-Hans' if tgt == 'zh' else tgt

    df = pd.read_excel(file).fillna('')
    batch_size = 200
    for i in range(0, df.shape[0], batch_size):
        sentences_src = list(df.iloc[i:i+batch_size, 0].values)
        sentences_bing = bing(sentences_src, src, tgt)
        df.loc[i:i+batch_size-1, "Microsoft Translator"] = sentences_bing
    df.to_excel(file, index=False, header=True)
    print("finished all", flush=True)


def main_bing3(file, src, tgt):
    src = 'zh-Hans' if src == 'zh' else src
    tgt = 'zh-Hans' if tgt == 'zh' else tgt

    sentences_src=open(file, 'r', encoding='utf8').readlines()
    sentences_src=[line.strip() for line in sentences_src]
    batch_size = 100
    sentences_bing=[]
    for i in range(0, len(sentences_src), batch_size):
        batch_sentences_src=sentences_src[i:i+batch_size]
        batch_sentences_bing = bing(batch_sentences_src, src, tgt)
        sentences_bing.extend(batch_sentences_bing)

    with open(file.replace("BBC_test.th","BBC_test_bing.en"), 'w') as fp:
        for item in sentences_bing:
            fp.write("{}\n".format(item))
    print("finished all", flush=True)


def async_itranslate(file):

    start_time = time.time()

    async def get_translation(session, sentence_src, src, tgt, i, df):
        async with session.post(url='https://dev-api.itranslate.com/translation/v2/',
                                headers={
                                    "Authorization": "Bearer 24784523-65ac-4c08-abcf-1ad90d398fe1",
                                    "Content-Type": "application/json"
                                },
                                json={
                                    "source": {"dialect": src, "text": sentence_src},
                                    "target": {"dialect": tgt}
                                }
                                ) as response:
            res = await response.json()
            sentence_tgt = res["target"]["text"]
            df.loc[i, "iTranslate"] = sentence_tgt
            return sentence_tgt

    async def _main():
        async with aiohttp.ClientSession() as session:
            tasks = []
            df = pd.read_excel(file)
            for i, sentence_src in enumerate(df.iloc[:, 0].values):
                tasks.append(asyncio.ensure_future(get_translation(session,
                                                                   sentence_src,
                                                                   "zh-CN",
                                                                   "en",
                                                                   i,
                                                                   df,
                                                                   )))
            original_pokemon = await asyncio.gather(*tasks, return_exceptions=True)
            print(i, flush=True)
            print(len(original_pokemon), flush=True)
            df.to_excel(file, index=False, header=True)
    asyncio.run(_main())
    print("--- %s seconds ---" % (time.time() - start_time))


def async_google(file):

    start_time = time.time()

    async def get_translation(session, sentences_src, src, tgt, i, df):
        async with session.post(url='https://translation.googleapis.com/language/translate/v2',
                                headers={
                                    "X-goog-api-key": "AIzaSyCCNyGLGl8F55oSDtIPWiigx9RJqolQDJE",
                                    "Content-Type": "application/json; charset=utf-8"
                                },
                                json={
                                    "q": sentences_src,
                                    "source": src,
                                    "target": tgt
                                }
                                ) as response:
            res = await response.json()
            sentences_tgt = [item['translatedText']
                             for item in res["data"]["translations"]]
            df.loc[i:i+99, "Google Translation"] = sentences_tgt
            return sentences_tgt

    async def _main():
        async with aiohttp.ClientSession() as session:
            tasks = []
            src = os.path.basename(file)[:2]
            tgt = os.path.basename(file)[3:5]
            df = pd.read_excel(file).fillna('')

            batch_size = 100
            for i in range(0, df.shape[0], batch_size):
                sentences_src = list(df.iloc[i:i+batch_size, 0].values)
                tasks.append(asyncio.ensure_future(get_translation(session,
                                                                   sentences_src,
                                                                   src,
                                                                   tgt,
                                                                   i,
                                                                   df,
                                                                   )))
            batches = await asyncio.gather(*tasks, return_exceptions=True)
            print(i, flush=True)
            print(len(batches), flush=True)
            df.to_excel(file, index=False, header=True)
    asyncio.run(_main())
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main_bing3("/home/xunlong/dataclean/cleaning/wukui/files/BBC_test.th", "th", "en")
