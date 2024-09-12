from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
import time
import torch
torch.cuda.set_device(0)

model_sentence_transformers = SentenceTransformer('./model/labse_bert_model')

MONGO_CONNECTION_STRING = "mongodb://localhost:27017/"
mongo_client = MongoClient(MONGO_CONNECTION_STRING)

db_data_pool = mongo_client['mlops']


def score_sentences(sentences_src, sentences_tgt):

    embeddings_src = model_sentence_transformers.encode(
        sentences_src, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    embeddings_tgt = model_sentence_transformers.encode(
        sentences_tgt, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    cosine_scores = util.dot_score(embeddings_src, embeddings_tgt)

    return cosine_scores


def clean_with_score(collection):

    sentences_src = []
    sentences_tgt = []
    ids = []

    results = db_data_pool[collection].find({'LaBSE': {'$exists': False}})

    for i, result in enumerate(results):
        sentences_src.append(result['sentence_src'])
        sentences_tgt.append(result['sentence_tgt'])
        ids.append(result['_id'])

        if (i+1) % 50000 == 0:
            cosine_scores = score_sentences(sentences_src, sentences_tgt)
            assert len(ids) == len(sentences_src) == len(
                sentences_tgt) == len(cosine_scores)
            for k in range(len(ids)):
                db_data_pool[collection].update_one({'_id': ids[k]},
                                                    {'$set': {'LaBSE': round(float(cosine_scores[k][k]), 4)}})
            sentences_src.clear()
            sentences_tgt.clear()
            ids.clear()
            print(k, flush=True)
            print(i, flush=True)

    cosine_scores = score_sentences(sentences_src, sentences_tgt)
    assert len(ids) == len(sentences_src) == len(
        sentences_tgt) == len(cosine_scores)
    for k in range(len(ids)):
        db_data_pool[collection].update_one({'_id': ids[k]},
                                            {'$set': {'LaBSE': round(float(cosine_scores[k][k]), 4)}})
    sentences_src.clear()
    sentences_tgt.clear()
    ids.clear()
    print(k, flush=True)
    print(i, flush=True)


def main(collection="en||ms"):

    start_time = time.time()

    clean_with_score(collection)
    print("finished {}".format(collection), flush=True)

    
    print("--- {} seconds ---".format(time.time() - start_time), flush=True)


if __name__ == '__main__':
    import plac
    plac.call(main)
