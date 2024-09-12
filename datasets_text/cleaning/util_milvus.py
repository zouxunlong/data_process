from sentence_transformers import SentenceTransformer, util
from milvus import Milvus, IndexType, MetricType, Status
from util_mongodb import db
import torch
import os

torch.cuda.set_device(0)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


_HOST = 'localhost'
_PORT = '19530'
_DIM = 768
_INDEX_FILE_SIZE = 2048

milvus = Milvus(host=_HOST, port=_PORT)

model_sentence_transformers = SentenceTransformer('all-mpnet-base-v2')


def milvus_query(collection_name, query, top_k):

    query_vector = model_sentence_transformers.encode([query],
                                                      show_progress_bar=False,
                                                      convert_to_numpy=True,
                                                      normalize_embeddings=True)
    search_param = {"nprobe": 100}

    param = {'collection_name': collection_name,
             'query_records': query_vector,
             'top_k': top_k,
             'params': search_param}

    status, results = milvus.search(**param)

    return results


def create_milvus_collection(collection_name):

    status, ok = milvus.has_collection(collection_name)
    if not ok:

        param = {'collection_name': collection_name,
                 'dimension': _DIM,
                 'index_file_size': _INDEX_FILE_SIZE,
                 'metric_type': MetricType.IP}

        milvus.create_collection(param)

    print(milvus.list_collections(), flush=True)
    print(milvus.get_collection_info(collection_name), flush=True)
    print(milvus.get_collection_stats(collection_name), flush=True)
    print(milvus.get_index_info(collection_name), flush=True)


def _insert(collection_name, vectors, milvus_ids):

    status, ids = milvus.insert(collection_name=collection_name,
                                records=vectors,
                                ids=milvus_ids)

    if not status.OK():
        print("Insert failed: {}".format(status), flush=True)
    else:
        milvus.flush([collection_name])

    print(milvus.get_collection_stats(collection_name), flush=True)
    print(milvus.get_index_info(collection_name), flush=True)


def mongo2milvus(collection_name):

    create_milvus_collection(collection_name)

    results = db[collection_name].find({}, {'sentence_src': 1, '_id': 1})
    sentences_src = []
    milvus_ids = []
    for item in results:
        sentences_src.append(item['sentence_src'])
        milvus_ids.append(item['_id'])

        if item['_id'] % 50000 == 0:

            embeddings_src = model_sentence_transformers.encode(
                sentences_src, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

            assert len(embeddings_src) == len(milvus_ids), "length of embeddings_src and ids don't match"

            _insert(collection_name, embeddings_src, milvus_ids)

            sentences_src.clear()
            # sentences_tgt.clear()
            milvus_ids.clear()
            print(item['_id'], flush=True)

    embeddings_src = model_sentence_transformers.encode(
        sentences_src, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    assert len(embeddings_src) == len(
        milvus_ids), "length of embeddings and ids don't match"

    _insert(collection_name, embeddings_src, milvus_ids)

    sentences_src.clear()

    milvus_ids.clear()

    print('all finished', flush=True)


def create_milvus_index(collection_name, nlist):
    index_param = {'nlist': nlist}
    status = milvus.create_index(collection_name,
                                 IndexType.IVF_FLAT,
                                 index_param)
    print('sucess: {}'.format(status.OK()))
    print(milvus.list_collections(), flush=True)
    print(milvus.get_collection_info(collection_name), flush=True)
    print(milvus.get_collection_stats(collection_name), flush=True)
    print(milvus.get_index_info(collection_name), flush=True)


def create_milvus_index(collection_name):
    print(milvus.drop_index(collection_name), flush=True)


if __name__ == "__main__":
    print(milvus.list_collections(), flush=True)

