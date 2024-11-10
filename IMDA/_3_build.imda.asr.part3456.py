import random
import traceback
import os
import re
from datasets import load_from_disk, Features, Value, Audio
import fire


instructions_asr = [
    "Can you help recognize the speech and transcribe it word for word?",
    "Please transcribe the content of this audio into text format.",
    "Could you convert the speech into a text transcript for me?",
    "Please transcribe."
]


def normalize_sentence(sentence):
    sentence = re.sub('<(tamil|malay|mandarin)>([^<>:]*):?([^<>:]*)</(tamil|malay|mandarin)>', r"\2", sentence)
    sentence = re.sub('(^|\s)<[a-zA-Z0-9]*>($|\s)', " ", sentence)
    sentence = re.sub('(^|\s)(\(ppb\)|\(ppc\)|\(ppl\)|\(ppo\))($|\s)', " ", sentence)
    sentence = re.sub('(^|\s)<[A-Z0-9]*/>($|\s)', " ", sentence)
    sentence = " ".join(re.sub('_', "", sentence).split()).strip()
    return sentence


def normalize_transcription(transcription, speaker):
    normalized_transcription = []
    for utterance in transcription:
        normalized_sentence = normalize_sentence(utterance["sentence"])
        if normalized_sentence:
            normalized_transcription.append(
                (speaker, normalized_sentence, utterance["start"], utterance["end"]))
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


def chunk(split, chunk_limit, workers=112):

    print("start {}".format(split), flush=True)
    partition = split.split("/")[-1]
    if os.path.exists(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF_ASR_{chunk_limit}/train/{partition}"):
        return
    ds = load_from_disk(split)
    if partition in ["PART3", "PART4", "PART5"]:
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
                "speaker1": ds.features["speaker1"],
                "speaker2": ds.features["speaker2"],
            }
        })
    else:
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
                fn_kwargs         = {"chunk_limit": chunk_limit, "partition": partition},
                batched           = True,
                batch_size        = 1,
                writer_batch_size = 1,
                features          = features,
                remove_columns    = ds.column_names,
                num_proc          = workers)
    ds_dict = ds.train_test_split(test_size=1000)
    ds_dict["train"].save_to_disk(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF_ASR_{chunk_limit}/train/{partition}")
    ds_dict["test"].save_to_disk(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF_ASR_{chunk_limit}/test/{partition}")


def main(
    chunk_limit,
    splits=[
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF/PART3",
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF/PART4",
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF/PART5",
        "/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/IMDA_HF/PART6"
    ]):
    for split in splits:
        chunk(split=split, chunk_limit=chunk_limit)
    print("complete all", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
