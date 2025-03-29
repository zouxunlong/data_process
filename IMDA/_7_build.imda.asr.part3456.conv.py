import random
import traceback
import os
import re
import unicodedata
from datasets import load_from_disk, Features, Value, Audio
import fire


instructions_asr = [
    "Can you help recognize the speech and transcribe it word for word?",
    "Please transcribe the content of this audio into text format.",
    "Could you convert the speech into a text transcript for me?",
    "Please transcribe."
]


def normalize_sentence(sentence):
    sentence = unicodedata.normalize('NFKC', sentence)
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = re.sub('<[a-zA-Z0-9/\s]*>', " ", sentence)
    sentence = re.sub('\((ppc|ppb|ppl|ppo)\)', " ", sentence, flags=re.IGNORECASE)
    sentence = re.sub('(_|\(|\)|\[|\])', "", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence


def normalize_transcription(transcription, speaker):
    normalized_transcription = []
    for utterance in transcription:
        normalized_sentence = normalize_sentence(utterance["sentence"])
        if normalized_sentence:
            normalized_transcription.append((speaker, normalized_sentence, utterance["start"], utterance["end"]))
    return normalized_transcription


def build_conversation(dialog):
    last_speaker = ""
    conversation = ""
    for speaker, sentence, start, end in dialog:
        if not last_speaker:
            conversation += f"{speaker}: {sentence}"
            last_speaker = speaker
        elif speaker == last_speaker:
            conversation += f" {sentence}"
        else:
            conversation += f"\n{speaker}: {sentence}"
            last_speaker = speaker
    return conversation


def swap_speakers(text):
    """Swaps speaker labels if speaker2 starts the conversation.
    Args:
        text: A string containing the conversation with speaker labels.
    Returns:
        A string with speaker labels potentially swapped.
    """
    lines = text.split('\n')
    if "<Speaker2>:" in lines[0]:
        swapped = True
    else:
        swapped = False
    if swapped:
        lines = [line.replace("<Speaker1>:", "speaker2:").replace("<Speaker2>:", "<Speaker1>:").replace("speaker2:", "<Speaker2>:") for line in lines]
    return swapped, "\n".join(lines)


def chunking(batch, chunk_limit, partition):
    try:
        new_batch = {
            "context"         : [],
            "instruction"     : [],
            "answer"          : [],
            "other_attributes": []
        }

        array          = batch["audio"][0]["array"]
        transcription1 = batch["transcription1"][0]
        transcription2 = batch["transcription2"][0]

        normalized_transcription1 = normalize_transcription(transcription1, "<Speaker1>")
        normalized_transcription2 = normalize_transcription(transcription2, "<Speaker2>")
        normalized_transcription  = normalized_transcription1 + normalized_transcription2

        normalized_transcription = [transcription
                                    for transcription in normalized_transcription
                                    if transcription[2] >= 0 and transcription[2] <= array.size/16000]
        normalized_transcription.sort(key=lambda item: item[2])

        dialogs = []
        dialog = []
        for speaker, sentence, start, end in normalized_transcription:
            if len(dialog) == 0:
                if end - start < chunk_limit:
                    dialog.append((speaker, sentence, start, end))

            elif end - dialog[0][2] < chunk_limit:
                dialog.append((speaker, sentence, start, end))

            else:
                if dialog[-1][3] - dialog[0][2] > chunk_limit/2:
                    dialogs.append(dialog.copy())
                dialog.clear()
                if end - start < chunk_limit:
                    dialog.append((speaker, sentence, start, end))

        for dialog in dialogs:
            start_time = dialog[0][2]
            end_time = dialog[-1][3]
            conversation = build_conversation(dialog)
            chunk_array = array[int(start_time*16000):int(end_time*16000)]

            swapped, text = swap_speakers(conversation)

            new_batch["context"].append({
                "text": None,
                "audio": {"array": chunk_array, "sampling_rate": 16000}
            })
            new_batch["instruction"].append({
                "text": random.choice(instructions_asr),
                "audio": None
            })
            new_batch["answer"].append({
                "text": text,
                "audio": None
            })
            if partition in ["PART3", "PART4", "PART5"]:
                if swapped:
                    new_batch["other_attributes"].append(
                        {
                            "conversation_id": batch["conversation_id"][0],
                            "start"          : start_time,
                            "end"            : end_time,
                            "setting"        : batch["setting"][0],
                            "partition"      : batch["partition"][0],
                            "speaker1"       : batch["speaker2"][0],
                            "speaker2"       : batch["speaker1"][0],
                        })
                else:
                    new_batch["other_attributes"].append(
                        {
                            "conversation_id": batch["conversation_id"][0],
                            "start"          : start_time,
                            "end"            : end_time,
                            "setting"        : batch["setting"][0],
                            "partition"      : batch["partition"][0],
                            "speaker1"       : batch["speaker1"][0],
                            "speaker2"       : batch["speaker2"][0],
                        })
            else:
                new_batch["other_attributes"].append(
                    {
                        "conversation_id": batch["conversation_id"][0],
                        "start": start_time,
                        "end": end_time,
                        "setting": batch["setting"][0],
                        "partition": batch["partition"][0],
                    })
        return new_batch
    except:
        print(traceback.format_exc(), flush=True)


def map2schema(ds, chunk_limit, part, workers=112):

    if part in ["PART3", "PART4", "PART5"]:
        features = Features({
            'context'         : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
            'instruction'     : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
            'answer'          : {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
            'other_attributes': {
                "conversation_id": ds.features["conversation_id"],
                "start"          : Value(dtype="float64"),
                "end"            : Value(dtype="float64"),
                "setting"        : ds.features["setting"],
                "partition"      : ds.features["partition"],
                "speaker1"       : ds.features["speaker1"],
                "speaker2"       : ds.features["speaker2"],
            }
        })
    if part in ["PART6"]:
        features = Features({
            'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
            'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
            'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
            'other_attributes': {
                "conversation_id": ds.features["conversation_id"],
                "start": Value(dtype="float64"),
                "end": Value(dtype="float64"),
                "setting": ds.features["setting"],
                "partition": ds.features["partition"],
            }
        })

    ds = ds.map(chunking,
                fn_kwargs         = {"chunk_limit": chunk_limit, "partition": part},
                batched           = True,
                batch_size        = 1,
                writer_batch_size = 1,
                features          = features,
                remove_columns    = ds.column_names,
                num_proc          = workers)
    return ds


def chunk(split, chunk_limit):

    part = split.split("/")[-1]
    if part == "PART3":
        test_conversation_ids={'2726', '3260', '3252', '2642', '3068', '3093', '3037', '3107', '3138'}
    if part == "PART4":
        test_conversation_ids={'0159', '0816', '1239', '0088', '2020', '0369', '0903', '0461', '0464', '0211', '0680', '0941', '0952', '0485', '0301'}
    if part == "PART5":
        test_conversation_ids={'3367_fin', '3447_pos', '3450_neg', '3250_pos', '4156_deb-1', '3100_pos', '3232_fin', '4198_deb-3', '3128_neg', '3177_neg', '3294_neg', '3313_neg', '3348_neg', '4137_deb-2', '3209_neg', '3459_pos', '3186_fin', '3504_fin', '3762_pos', '4219_deb-3', '3540_fin', '3388_pos', '4168_deb-2', '4128_deb-2', '3104_neg', '3548_pos', '4041_deb-3', '3759_fin', '3004_fin', '4026_deb-2', '4082_deb-1', '3237_pos', '3083_pos', '4067_deb-3', '3433_pos', '3226_pos', '4242_deb-1', '4224_deb-2', '3095_pos', '3277_pos', '4187_deb-2', '4079_deb-2', '3723_pos', '3370_pos', '4069_deb-3', '3606_neg', '3178_neg', '3733_fin', '3773_pos', '3523_fin', '3642_fin', '3718_pos', '3199_pos', '3068_neg', '3695_pos', '3416_pos', '3321_fin', '4091_deb-3', '3229_fin'}
    if part == "PART6":
        test_conversation_ids={'0369_cc-hol', '1130_cc-bnk', '0302_cc-hol', '0247_cc-res', '1340_cc-bnk', '1436_cc-hdb', '1803_cc-hdb', '0266_cc-res', '0106_cc-hol', '0169_cc-hol', '0392_cc-res', '1123_cc-ins', '0488_cc-res', '0893_cc-tel', '1286_cc-bnk', '1816_cc-hdb', '0365_cc-res', '1968_cc-msf', '1831_cc-moe', '0927_cc-bnk', '1751_cc-msf', '0144_cc-hol', '1000_cc-ins', '0508_cc-res', '1342_cc-tel', '1776_cc-moe', '1292_cc-ins', '1692_cc-hdb', '1989_cc-moe', '1371_cc-msf', '0879_cc-ins', '1188_cc-ins', '0439_cc-hot', '0297_cc-hot', '0877_cc-bnk', '0550_cc-hot', '1422_cc-moe', '0573_cc-hot', '0681_cc-hol', '1308_cc-bnk', '1911_cc-hdb', '1960_cc-hdb', '0152_cc-hol', '1535_cc-moe', '0801_cc-ins', '0942_cc-bnk', '1546_cc-moe', '1336_cc-bnk', '1447_cc-hdb', '0177_cc-hot', '0129_cc-res', '1481_cc-moe', '1316_cc-tel', '0136_cc-hol', '1152_cc-bnk', '1071_cc-ins', '0420_cc-res', '1949_cc-moe', '0627_cc-hol', '1982_cc-msf', '1745_cc-hdb', '1267_cc-tel', '1644_cc-moe', '0620_cc-hot', '1626_cc-msf', '0983_cc-ins', '1522_cc-moe', '1496_cc-msf', '0049_cc-hol', '0304_cc-hol', '1882_cc-moe', '0973_cc-ins', '1584_cc-msf', '1542_cc-moe', '0269_cc-hot', '1164_cc-ins', '1283_cc-bnk', '0244_cc-hot', '0970_cc-bnk', '0751_cc-ins', '0172_cc-res', '0268_cc-hot', '0069_cc-res', '1670_cc-msf', '1568_cc-moe', '1466_cc-msf', '1051_cc-ins', '1684_cc-hdb', '1663_cc-moe', '0728_cc-tel', '1903_cc-moe', '1441_cc-msf', '1892_cc-msf', '1028_cc-tel', '1957_cc-hdb', '0416_cc-res', '0920_cc-tel', '0409_cc-res', '1966_cc-hdb', '2016_cc-moe', '0700_cc-ins', '1528_cc-msf', '0649_cc-hol', '0944_cc-bnk', '0215_cc-hot', '2032_cc-hdb', '0708_cc-bnk', '1756_cc-hdb', '1252_cc-ins', '0572_cc-hol', '1975_cc-hdb', '0610_cc-hot', '1687_cc-hdb', '0074_cc-hol', '0454_cc-hol', '0858_cc-bnk', '1460_cc-hdb', '0643_cc-hol', '1759_cc-hdb'}

    ds       = load_from_disk(split)
    ds_test  = ds.filter(lambda x: [item in test_conversation_ids for item in x["conversation_id"]], batched=True, num_proc=4)
    ds_train = ds.filter(lambda x: [item not in test_conversation_ids for item in x["conversation_id"]], batched=True, num_proc=4)

    ds_test = map2schema(ds_test, chunk_limit, part)
    print("ds_test: ", len(ds_test), flush=True)
    ds_test.save_to_disk(f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_asr/test/ASR/IMDA_{part}_{chunk_limit}_ASR", num_proc=10)

    ds_train = map2schema(ds_train, chunk_limit, part)
    print("ds_train: ", len(ds_train), flush=True)
    ds_train.save_to_disk(f"/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_asr/train/ASR/IMDA_{part}_{chunk_limit}_ASR", num_proc=10)


def main():
    splits=[
        # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_conv_hf/PART3",
        # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_conv_hf/PART4",
        # "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_conv_hf/PART5",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_conv_hf/PART6",
    ]
    for split in splits:
        for chunk_limit in [30, 60, 120, 300]:
            print("start {} chunk_limit: {}".format(split, chunk_limit), flush=True)
            chunk(split=split, chunk_limit=chunk_limit)
    print("complete all", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
