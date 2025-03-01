import time
from sentence_transformers import SentenceTransformer, util


model_sentence_transformers = SentenceTransformer('labse_bert_model')


def embedding_saving(sentences_src, sentences_tgt, file_path_out, pool):

    source_embedding = model_sentence_transformers.encode_multi_process(
        sentences_src, pool)

    target_embedding = model_sentence_transformers.encode_multi_process(
        sentences_tgt, pool)

    assert len(source_embedding) == len(
        target_embedding), "length of src and target don't match"

    cosine_scores = util.cos_sim(source_embedding, target_embedding)

    with open(file_path_out, 'a', encoding='utf8') as f_out:
        for k in range(len(cosine_scores)):
            cosine_score = cosine_scores[k][k]
            f_out.write("{:.4f} ||| {} ||| {}\n".format(
                cosine_score, sentences_src[k].replace("|||", " "), sentences_tgt[k].replace("|||", " ")))


def clean_with_score(file_path_src, file_path_tgt, file_path_out, pool):
    with open(file_path_src, encoding='utf8') as file_src, \
            open(file_path_tgt, encoding='utf8') as file_tgt:

        sentences_src = []
        sentences_tgt = []

        for (i, sentence_src), (j, sentence_tgt) in zip(enumerate(file_src), enumerate(file_tgt)):
            if len(sentence_src.strip()) > 15 and len(sentence_tgt.strip()) > 15:
                sentences_src.append(sentence_src.strip())
                sentences_tgt.append(sentence_tgt.strip())

            if (i+1) % 50000 == 0:
                embedding_saving(sentences_src, sentences_tgt,
                                 file_path_out, pool)
                sentences_src.clear()
                sentences_tgt.clear()
                print(i, flush=True)

        embedding_saving(sentences_src, sentences_tgt, file_path_out, pool)

        print("finished {}".format(i), flush=True)


def main():

    start_time = time.time()

    pool = model_sentence_transformers.start_multi_process_pool()

    clean_with_score('/home/xuanlong/dataclean/cleaning/data/V4.en',
                     '/home/xuanlong/dataclean/cleaning/data/V4.th',
                     '/home/xuanlong/dataclean/cleaning/data/V4.en-th', pool)

    model_sentence_transformers.stop_multi_process_pool(pool)
    
    print("finished", flush=True)
    print("--- {} seconds ---".format(time.time() - start_time), flush=True)

if __name__ == '__main__':
    main()
