from pymongo import MongoClient
from custom_analyzers import ThAnalyzer, ViAnalyzer, TaAnalyzer
from whoosh.analysis import StemmingAnalyzer, SimpleAnalyzer
from jieba.analyse import ChineseAnalyzer
from pymongo import MongoClient
import json
import pandas as pd


MONGODB_CONNECTION_STRING = 'mongodb://localhost:27017/'
mongo_client = MongoClient(MONGODB_CONNECTION_STRING)


db = mongo_client['mlops']


def get_analyzer(lang):
    if lang == 'en':
        analyzer = StemmingAnalyzer()
    elif lang == 'zh':
        analyzer = ChineseAnalyzer()
    return analyzer


def insert_data_0(collection_name, df):
    try:
        for i, [index,
                edited_sentence_src,
                edited_sentence_tgt,
                lang_src,
                lang_tgt,
                domain,
                _id] in enumerate(df.loc[0:].values):

            result = db[collection_name].update_one({'_id': _id}, {'$set': {'edited_sentence_src': edited_sentence_src.strip(),
                                                                            'edited_sentence_tgt': edited_sentence_tgt.strip(),
                                                                            'lang_src': lang_src,
                                                                            'lang_tgt': lang_tgt,
                                                                            'domain': domain
                                                                            }}, upsert=True)

        print("finished insert {} documents.".format(i+1), flush=True)
    except Exception as err:
        print(err, flush=True)


def insert_data_1(file_path_src, file_path_tgt, lang_src, lang_tgt, collection):
    try:
        analyzer_src = get_analyzer(lang_src)
        analyzer_tgt = get_analyzer(lang_tgt)
        n = 10000000000001
        with open(file_path_src, encoding='utf8') as file_src,\
                open(file_path_tgt, encoding='utf8') as file_tgt:
            for (i, line_src), (j, line_tgt) in zip(enumerate(file_src), enumerate(file_tgt)):
                tokens_src = [
                    token.text for token in analyzer_src(line_src.strip())]
                tokens_tgt = [
                    token.text for token in analyzer_tgt(line_tgt.strip())]
                result = collection.insert_one({'_id': n+i,
                                                'sentence_src': line_src.strip(),
                                                'sentence_tgt': line_tgt.strip(),
                                                'lang_src': lang_src,
                                                'lang_tgt': lang_tgt,
                                                'tokens_src': tokens_src,
                                                'tokens_tgt': tokens_tgt,
                                                'domain': ['full-domain']})

            print("finished insert {} documents.".format(i+1), flush=True)
    except Exception as err:
        print(err, flush=True)
        print(i, flush=True)


def insert_data_2(jl_path, lang_src, lang_tgt, collection):
    try:
        analyzer_src = get_analyzer(lang_src)
        analyzer_tgt = get_analyzer(lang_tgt)
        n = 10000000000001
        with open(jl_path, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                item = json.loads(line)
                tokens_src = [
                    token.text for token in analyzer_src(item["sentence_src"].strip())]
                tokens_tgt = [
                    token.text for token in analyzer_tgt(item["sentence_tgt"].strip())]
                result = collection.insert_one({'_id': n+i,
                                                'sentence_src': item["sentence_src"].strip(),
                                                'sentence_tgt': item["sentence_tgt"].strip(),
                                                'source_lang': item["source_lang"],
                                                'target_lang': item["target_lang"],
                                                'tokens_src': tokens_src,
                                                'tokens_tgt': tokens_tgt,
                                                'domain': ['full-domain', 'news']})

            print("finished insert {} documents.".format(i+1), flush=True)
    except Exception as err:
        print(err, flush=True)


def build_index_text(collection):

    collection.create_index(
        [
            ('tokens_tgt', 1),
        ]
    )
    collection.create_index(
        [
            ('tokens_src', 1),
        ]
    )



def search_query(query, lang, collection_name='wukui'):

    analyzer = get_analyzer(lang)

    tokens_query = [token.text for token in analyzer(query)]

    print(tokens_query, flush=True)

    pipeline = []
    pipeline.append({"$match": {"$or": [{"$and": [{"lang_src": lang}, {"tokens_src": {"$all": tokens_query}}]},
                                        {"$and": [{"lang_tgt": lang}, {"tokens_tgt": {"$all": tokens_query}}]}]}})

    pipeline.append({"$limit": 20000})

    results = db[collection_name].aggregate(pipeline)

    retrieved_items = [result for result in results]
    items_df = pd.DataFrame.from_records(data=retrieved_items, columns=['sentence_src',
                                                                        'sentence_tgt',
                                                                        'lang_src',
                                                                        'lang_tgt',
                                                                        'domain',
                                                                        '_id'])
    if not items_df.empty:
        items_df['_id'] = items_df['_id'].apply(lambda s: str(s))
        items_df = items_df.reset_index()
    return items_df, tokens_query



def search_query2(query, lang, collection_name='wukui'):

    print(query, flush=True)

    pipeline = []
    pipeline.append({"$match": {'$text': {'$search': query}}})
    pipeline.append({"$sort": {"score": {'$meta': "textScore"}}})
    pipeline.append({"$limit": 20000})

    results = db[collection_name].aggregate(pipeline)

    retrieved_items = [result for result in results]
    items_df = pd.DataFrame.from_records(data=retrieved_items, columns=['sentence_src',
                                                                        'sentence_tgt',
                                                                        'lang_src',
                                                                        'lang_tgt',
                                                                        'domain',
                                                                        '_id'])
    if not items_df.empty:
        items_df['_id'] = items_df['_id'].apply(lambda s: str(s))
        items_df = items_df.reset_index()
    return items_df, query



def search_ids(ids, collection_name='wukui'):

    pipeline = []
    pipeline.append({"$match": {"_id": {"$in": ids}}})
    pipeline.append(
        {"$addFields": {"__order": {"$indexOfArray": [ids, "$_id"]}}})
    pipeline.append({"$sort": {"__order": 1}})
    pipeline.append({"$limit": 20000})

    results = db[collection_name].aggregate(pipeline)

    retrieved_items = [result for result in results]
    items_df = pd.DataFrame.from_records(data=retrieved_items, columns=['sentence_src',
                                                                        'sentence_tgt',
                                                                        'lang_src',
                                                                        'lang_tgt',
                                                                        'domain',
                                                                        '_id'])

    if not items_df.empty:
        items_df['_id'] = items_df['_id'].apply(lambda s: str(s))
        items_df = items_df.reset_index()
    return items_df


if __name__ == "__main__":
    search_query("this is singapore's best food", "en")
    print("finished", flush=True)
