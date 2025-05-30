
from multiprocessing import Pool
from datasets import load_from_disk, Audio, Features, Value, concatenate_datasets
import random, os
from fire import Fire
from tqdm import tqdm


instructions_asr = [
    "Please transcribe this audio.",
    "Convert this speech to text format.",
    "Document the spoken dialogue.",
    "Record the verbal exchange in written form.",
    "Transform the oral presentation into a text document.",
    "Capture the spoken words as written text.",
    "Translate the audio recording into text.",
    "Write out the speech from this audio.",
    "Provide a written transcript of these words.",
    "Generate a text version of this spoken content.",
    "Document the audio content in text.",
    "Transcribe the dialogue from this recording.",
    "Convert the vocal expressions to written words.",
    "Transform these spoken terms into text.",
    "Create a document from this spoken audio.",
    "Render the oral communication into text.",
    "Produce a written record of this speech.",
    "Translate this vocal recording into a textual format.",
    "Transcribe this sound recording to text.",
    "Put this spoken language into written form.",
    "Transcribe the spoken segments of this audio file.",
    "Transform the vocal sounds into readable text.",
    "Convert the voice recording to written language.",
    "Write down what's being said in this audio.",
    "Produce a textual transcript of the oral remarks.",
    "Transcribe the contents of this voice file.",
    "Detail the spoken words in text form.",
    "Convert the speech sounds into text.",
    "Transcribe this audio clip to text.",
    "Transform this voice note into written words.",
    "Provide a text translation of this spoken dialogue.",
    "Record the audio conversation in text.",
    "Create a written version of these spoken words.",
    "Generate a textual document from this audio.",
    "Document the speech in this recording as text.",
    "Write out the contents of this sound file.",
    "Produce text from spoken words in this recording.",
    "Convert spoken dialogue into text documentation.",
    "Translate this audio file into written text.",
    "Turn the spoken parts of this clip into text.",
    "Provide a transcription of the spoken audio.",
    "Document this spoken session in written format.",
    "Transform the audio speech into a written transcript.",
    "Produce a written interpretation of this speech.",
    "Transcribe the words spoken in this audio.",
    "Translate spoken content into a text file.",
    "Create a written transcript from this voice recording.",
    "Convert the oral speech into a textual representation.",
    "Transform this sound bite into written text.",
    "Capture the spoken audio in text format.",
    "Render this voice message into text.",
    "Generate a written output from this spoken input.",
    "Provide a text version of the voice content.",
    "Transcribe the audible communication into written form.",
    "Create a textual record of the spoken communication.",
    "Turn the voice data into a text document.",
    "Produce a written copy of this audio dialogue.",
    "Document the verbal content in text.",
    "Write a transcription of the spoken phrases.",
    "Transform the oral narrative into written words.",
    "Convert this voice into a written transcript.",
    "Provide a written form of the audio speech.",
    "Record this speech as text.",
    "Turn the spoken language into a text format.",
    "Generate a document from the audio speech.",
    "Translate the spoken word into text form.",
    "Create a text representation of the spoken content.",
    "Document the voice recording in text.",
    "Transcribe the audio into a written form.",
    "Convert this spoken segment to text.",
    "Transform spoken expressions into written text.",
    "Provide a textual summary of this spoken audio.",
    "Write out the verbal content of this recording.",
    "Turn the audio speech into readable text.",
    "Produce a document reflecting the spoken words.",
    "Translate the spoken audio into written documentation.",
    "Create a transcript from this spoken content.",
    "Record the spoken words in a text file.",
    "Produce a text from the audio content.",
    "Transform the voice clip into written language.",
    "Document the oral content as text.",
    "Write down the audio conversation.",
    "Transcribe the spoken information into text.",
    "Convert the audible parts of this recording into text.",
    "Provide a written translation of the oral words.",
    "Document the verbal audio in written form.",
    "Create text from this spoken recording.",
    "Turn spoken words into a text document.",
    "Generate a text output from spoken input.",
    "Transcribe the oral conversation into written text.",
    "Convert the audio speech to written words.",
    "Create a written document from this oral communication.",
    "Record the spoken word in text form.",
    "Transform this spoken presentation into text.",
    "Transcribe the voice to text.",
    "Translate this recorded speech into text.",
    "Document this voice conversation as text.",
    "Write out the spoken dialogue as text.",
    "Generate a textual transcription of the audio conversation.",
    "Create a written account of the spoken words in this recording.",
    "Convert the spoken words to written text.",
    "Transcribe the verbal content into a document.",
    "Please create a text transcription of the audio.",
    "Kindly convert the audio into text.",
    "Please write down the contents of the audio.",
    "Please document the spoken words as text.",
    "Turn the spoken language into written form.",
    "Produce a text version of the spoken content.",
    "Transcribe the audio recording into text.",
    "Document the contents of this audio in written form.",
    "Capture the audio in written text.",
    "Render the spoken audio into written words.",
    "Convert the audio file to a textual document.",
    "Provide a written transcription of the audio.",
    "Translate the spoken words into text format.",
    "Record the audio content as written text.",
    "Write out the audio into a text file.",
    "Generate a text document from this audio.",
    "Transform the audio speech into text.",
    "Create a written record of the audio.",
    "Convert these spoken words into a text format.",
    "Please produce a text copy of this sound recording.",
    "Transcribe this audio into written words.",
    "Make a textual transcription of the spoken audio.",
    "Document the verbal audio into text.",
    "Convert the audio speech into a text transcript.",
    "Transcribe the spoken words into written form.",
    "Listen to the audio and provide the text version.",
    "Transform the speech into a text document.",
    "Capture the spoken language and convert it to text.",
    "Decode the audio and give me the written transcript.",
    "Recognize the verbal communication and transcribe it into text.",
    "Turn the vocal input into a text transcription.",
    "Process the audio speech and provide the text output.",
    "Translate the spoken conversation into written text.",
    "Transcribe the spoken dialogue into written format.",
    "Convert this audio recording to a text document.",
    "Please provide a written transcription of the speech.",
    "Write out the spoken words from the audio file.",
    "Transcribe the audio clip into text.",
    "Convert the voice message into a text transcript.",
    "Generate a text version of this spoken conversation.",
    "Turn the recorded speech into a text file.",
    "Write down the words spoken in this recording.",
    "Transcribe the spoken audio into readable text.",
    "Convert the verbal presentation into written form.",
    "Turn the spoken narrative into a textual document.",
    "Translate this oral speech into a written transcript.",
    "Provide a text transcription of the recording.",
    "Transcribe the verbal discussion into a document.",
    "Write out the dialogue as text.",
    "Convert spoken sound to written transcripts.",
    "Transcribe the audio from the podcast into text.",
    "Provide a written version of the audio.",
    "Write out the spoken words from the news clip.",
    "Provide a text version of the radio broadcast.",
    "Write out the spoken words from the documentary.",
    "Provide a transcription of the oral recording.",
    "Turn the audio into a detailed text document.",
    "Transcribe the informal chat into a text transcript.",
    "Can you type out what's in this audio?",
    "Turn this talk into text, will ya?",
    "Need this spoken bit written down, please.",
    "Help me get this convo in writing.",
    "Can you jot down what they’re saying here?",
    "Mind putting this chat into words?",
    "Write up what’s being said in this clip, okay?",
    "Got a minute to transcribe this audio?",
    "Let’s get this speech on paper.",
    "Could you write out this audio for me?",
    "Make a doc out of this talk, please.",
    "Turn this chit-chat into text, will you?",
    "I need this recorded talk in text form.",
    "Capture what they're saying in writing.",
    "Can you document this talk for me?",
    "Transcribe this, would you?",
    "Type up this convo for me, thanks!",
    "Get me a written version of this dialogue.",
    "Could you help me write this audio down?",
    "Write this speech out, please.",
    "Convert this talking into a text file, okay?",
    "Can you make this audio into text?",
    "Need this speech typed out, like now.",
    "Help put this voice recording in words.",
    "Can you document what he's saying?",
    "Jot down this voice note for me, please.",
    "Write out what’s in this voice clip?",
    "Turn this sound into text for me.",
    "Make a text doc from this audio, alright?",
    "Could you type this spoken stuff into text?",
    "Help me turn this chat into a document.",
    "Need these words on paper, can you do that?",
    "Can you transcript this talk for me?",
    "Get this audio into text form, will ya?",
    "Put this conversation in text, please.",
    "Type out what they're talking about?",
    "Mind converting this audio to text?",
    "Can you write out this conversation?",
    "Capture this audio in text, please.",
    "Write down what's spoken in this audio, thanks!",
    "Help me document this talk in text.",
    "Turn this audio into a text note?",
    "Need a text version of this voice message.",
    "Can you get this speech into writing?",
    "Make a text copy of this chat, please.",
    "Could you type out this voice recording?",
    "Let’s put this conversation on paper.",
    "Can you make a written note of this speech?",
    "Write this down from the audio, okay?",
    "Help transcribe this, would you?",
    "Mind writing out this speech for me?",
    "Can you make a doc from this audio?",
    "Document this convo in a text file.",
    "Write out what they're saying here.",
    "Turn this talk into a written piece.",
    "Need this conversation in text, can you help?",
    "Can you turn this speech into text?",
    "Mind making a text file from this audio?",
    "Jot this audio down in text for me?",
    "Write up this chat into text, please.",
    "Can you transcribe what’s being said?",
    "Turn this dialogue into text for me.",
    "Let’s convert this chat into writing.",
    "Need this talk written down, please.",
    "Type up this audio for me.",
    "Capture this voice recording in text, alright?",
    "Document this speech into text, will ya?",
    "Can you make a text transcript of this?",
    "Help me put this conversation in writing.",
    "Type out this dialogue from the audio.",
    "Turn this voice into text, can you?",
    "Make a written record of this audio.",
    "Document this chat in text, please.",
    "Capture what's in this audio, alright?",
    "Can you put this talk into written words?",
    "Write down what this audio says.",
    "Turn this speech recording into text.",
    "Mind converting this talk to text?",
    "Write up what’s spoken here, okay?",
    "Transcribe this voice clip for me, thanks.",
    "Can you get this talk in text?",
    "Help me make a text out of this audio.",
    "Type this spoken bit into a document.",
    "Jot down what they're saying in this recording.",
    "Turn this audio talk into text.",
    "Make a written form of this voice chat.",
    "Need a transcript of this conversation.",
    "Can you document this audio in text?",
    "Write this voice recording into a text doc.",
    "Capture what's spoken in this clip, will you?",
    "Type out this voice to text, okay?",
    "Make a text version of this spoken word.",
    "Can you put this audio into words?",
    "Help transcribe this speech into text.",
    "Document what's said in this audio.",
    "Turn this chat into a text document.",
    "Write up what's in this recording, please.",
    "Convert this speech to a text file.",
    "Can you write down what’s spoken here?",
    "Need this audio turned into a written piece.",
]


def get_all_split(root_hf):
    directories = []
    for dirpath, dirs, files in os.walk(root_hf):
        if len(dirs) == 0:
            directories.append(dirpath)
    return directories


def map_fn(example):
    return {
        "context": {
            "text": None,
            "audio": example["audio"]
        },
        "instruction": {
            "text": random.choice(instructions_asr),
            "audio": None
        },
        "answer": {
            "text": example["text"],
            "audio": None
        },
        "other_attributes": {
            "id": example["id"],
            "duration_ms": example["duration_ms"],
        }
    }


def map2schema(split, workers=120):

    ds = load_from_disk(split)

    features = Features({
        'context': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'instruction': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'answer': {"text": Value(dtype='string'), "audio": Audio(sampling_rate=16000, decode=True)},
        'other_attributes': {
            "id": ds.features["id"],
            "duration_ms": ds.features["duration_ms"],
        }
    })

    num_samples=len(ds)
    print("num_samples: ", num_samples, flush=True)

    for batch, i in enumerate(range(0, num_samples, 24000)):

        ds_slice = ds.select(range(i, min(i+24000, num_samples)))
        ds_slice = ds_slice.map(map_fn,
                                features             = features,
                                remove_columns       = ds.column_names,
                                num_proc             = workers,
                                batch_size           = 1,
                                writer_batch_size    = 1,
                                desc                 = f"mapping {i}-{min(i+24000, num_samples)}"
                                )
        problem_ids=[]
        for i in tqdm(range(len(ds_slice)), desc = f"filtering {i}-{min(i+24000, num_samples)}"):
            try:
                sample=ds_slice[i]
            except:
                problem_ids.append(i)
        ds_slice = ds_slice.select([i for i in range(len(ds_slice)) if i not in problem_ids])
        ds_slice.save_to_disk(f"{split}_v1/{batch}", num_proc=4)    
        print(f"complete saving {split}_v1/{batch}", flush=True)


def main(dir):
    splits = get_all_split(dir)
    for split in splits:
        print("start {}".format(split), flush=True)
        map2schema(split)
        print("complete {}".format(split), flush=True)


if __name__ == "__main__":
    Fire(main)
