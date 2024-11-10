
import os
from datasets import load_from_disk
import random
from openai import OpenAI
import random


candidate_instructions = [
    "Create a brief summary capturing the key points and decisions made during the dialogue.",
    "Condense the dialogue into a concise summary highlighting major topics and conclusions.",
    "Write a short overview focusing on the main discussions and outcomes of the dialogue.",
    "Provide a succinct summary that captures the essence and key outcomes of the dialogue.",
    "Summarize the dialogue by focusing on critical points and speaker contributions.",
    "Draft a concise summary that includes main topics and any decisions reached.",
    "Outline the primary points and conclusions discussed in the dialogue.",
    "Generate a summary that distills the dialogue into its most important discussions and decisions.",
    "Compose a brief summary emphasizing the key topics and decisions within the dialogue.",
    "Construct a concise encapsulation of the dialogue, noting significant topics and outcomes.",
    "Develop a summary that succinctly captures the essential points and results of the dialogue.",
    "Formulate a summary that condenses the main discussions and resolutions of the dialogue.",
    "Produce a short summary detailing the primary discussions and decisions from the dialogue.",
    "Prepare a concise overview of the dialogue, highlighting central themes and decisions.",
    "Craft a summary that focuses on the pivotal points and conclusions of the dialogue.",
    "Assemble a brief recap of the dialogue, emphasizing important topics and outcomes.",
    "Forge a succinct narrative summarizing the dialogue’s main points and resolutions.",
    "Synthesize the dialogue into a concise summary, focusing on key discussions and decisions.",
    "Compile a concise report of the dialogue, spotlighting major topics and outcomes.",
    "Create a streamlined summary that captures the core discussions and decisions of the dialogue."
]


def map_fn(sample):

    port = random.choice([8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007])
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )

    # Dialog summarization
    transcription = sample['answer']['text']

    TEMPLATE = """\
        [Speech Transcription]
        {context}

        [Task]
        You are given one dialogue. Please summarize the main points discussed, focusing on key topics.
        Mention the speakers by name and highlight any specific contributions or statements they made that are critical to understanding the dialogue.
        Ensure your summary captures the essence and provides a clear overview of the dialogue without introducing any assumptions or interpretations not evident in the transcription.

        Format your response as follows:
        Summary: (Provide a concise summary of the dialogue here.)
        """

    format_sample = TEMPLATE.format(context=transcription)

    chat_response = client.chat.completions.create(
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        messages=[{"role": "user", "content": format_sample}]
    )

    output = chat_response.choices[0].message.content

    if "Summary:" in output:
        summary = output.split('Summary:')[1].strip()
    else:
        summary = "No match found!!!!"

    sample["instruction"]["text"] = random.choice(candidate_instructions)
    sample["answer"]["text"] = summary
    sample["other_attributes"]["transcription"] = transcription
    breakpoint()
    return sample


for ROOT_PATH in ['/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal/test/ASR',
                  '/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal/train/ASR']:

    for DATASET_NAME in [
        'IMDA_PART3_30_ASR_v4',
        'IMDA_PART4_30_ASR_v4',
        'IMDA_PART5_30_ASR_v4',

        'IMDA_PART3_60_ASR_v4',
        'IMDA_PART4_60_ASR_v4',
        'IMDA_PART5_60_ASR_v4',

        'IMDA_PART3_120_ASR_v4',
        'IMDA_PART4_120_ASR_v4',
        'IMDA_PART5_120_ASR_v4',
    ]:

        data = load_from_disk(os.path.join(ROOT_PATH, DATASET_NAME))

        data = data.filter(lambda x: len(x['answer']['text'].strip().split()) > 8,
                           batch_size        = 1,
                           writer_batch_size = 1,
                           num_proc          = 224
                           )
    
        data = data.map(map_fn,
                        num_proc=224,
                        batch_size=1,
                        writer_batch_size=1,
                        )

        data = data.filter(lambda x: x['answer']['text'].strip() not in ['No match found!!!!', ''],
                            num_proc=224,
                            batch_size=1,
                            writer_batch_size=1,
                            )

        data.save_to_disk(os.path.join(ROOT_PATH.replace("/ASR", "/DS"), DATASET_NAME.replace("_ASR_","_DS_")), num_proc=2)


