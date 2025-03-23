from datasets import load_from_disk

ds=load_from_disk("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/from_zh/gigaspeech_ASR_v2")

def map_fn(answer, other_attributes):
    transcription = other_attributes[0]["transcription"]
    
    if "<" in transcription:
        print(transcription, flush=True)
        breakpoint()
    return {}
    
    # if not "'" in transcription:
    #     return {}

    # transcription = re.sub('<[a-zA-Z0-9/\s]*>', "", transcription)
    # terms         = transcription.split(" ")

    # mappings = {}
    # for i, term in enumerate(terms):
    #     if "'" in term:
    #         phrase = " ".join(terms[i:i+2])
    #         mappings[phrase.replace("'", "")] = phrase

    # answer = answer[0]["text"]
    # for key, value in mappings.items():
    #     match = re.findall(key, answer, flags=re.IGNORECASE)
    #     print(match)

    # return {"answer": [answer]}


def restore_apostrophes(original, processed, punc_to_restore = "'-", punc_to_skip = "_=+,.!?;:\'\"\\| "):
    """
    Restore apostrophes in the processed string by comparing it with the original.
    Assumes that the only difference is the missing apostrophes (and possible extra punctuation).
    """
    result = []
    i, j = 0, 0
    while i < len(original) and j < len(processed):
        if original[i].lower() == processed[j].lower():
            result.append(processed[j])
            i += 1
            j += 1
        else:
            # If the character in the original is an apostrophe,
            # add it to the result and advance the pointer in the original string.
            if original[i] in punc_to_restore:
                result.append(original[i])
                i += 1
            else:
                # Otherwise, just copy the character from processed.
                if original[i] in punc_to_skip:
                    i += 1
                if processed[j] in punc_to_skip:
                    result.append(processed[j])
                    j += 1
 
    # If there are any remaining characters in processed (e.g. trailing punctuation), append them.
    if j < len(processed):
        result.append(processed[j:])
    # Also, if the original still has remaining apostrophes, append those.
    if i < len(original):
        for ch in original[i:]:
            if ch == "'":
                result.append(ch)
    return "".join(result)

def map_fn2(answer, other_attributes):
    transcription = other_attributes[0]["transcription"]
    if not ("'" in transcription or "-" in transcription):
        return {}
    answer_text = answer[0]["text"]
    transcription = transcription.replace("<PERIOD>", ".").replace("<COMMA>", ",").replace("<QUESTIONMARK>", "?").replace("<EXCLAMATIONPOINT>", "!")
    new_answer_text=restore_apostrophes(transcription, answer_text)

    return {"answer": [{"audio":None, "text":new_answer_text}]}
    
ds=ds.map(map_fn2,
          batched=True,
          batch_size=1,
          writer_batch_size=1,
          input_columns=["answer", "other_attributes"],
          num_proc=224)

ds.save_to_disk("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/from_zh/gigaspeech_ASR_v4", num_proc=4)
