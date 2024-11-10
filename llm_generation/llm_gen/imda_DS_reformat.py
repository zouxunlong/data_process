

import os
import random

import fire
from openai import OpenAI
from datasets import load_from_disk


# os.environ["HF_HOME"] = "~/scratch/huggingface"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# HF_HOME=~/scratch/huggingface HF_ENDPOINT=https://hf-mirror.com python imda_accent_reformat.py 2>&1 | tee imda_accent_reformat.log


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


def map_fn(batch):

    prompt = """

        You're give 3 examples to generate a summary based on the transcription of an audio clip.
        Please summarize the main points discussed, focusing on key topics.
        Mention the speakers by name and highlight any specific contributions or statements they made that are critical to understanding the dialogue.
        Ensure your summary captures the essence and provides a clear overview of the dialogue without introducing any assumptions or interpretations not evident in the transcription.
        Do not output anything else except the summarization.

        Example 1:
        Transcription: <Speaker1>: ya it's no no seats [ah] we went all the way to the front I was like I was like I was like I was like this close like about three metres away from Jay-#Chou# I was like if I stretched out I could have touched him you know ya like like that's that's how close and then that was like one of the best thing that happened like I really like Jay-#Chou# I think is very interesting\n<Speaker2>: I get touch him already\n<Speaker1>: (uh) he's a very good singer next to JJ-#Lin# [lah] most people like JJ-#Lin# more in Singapore but I think Jay-#Chou# is (um) like his song better\n<Speaker2>: ya I think Jay-#Chou# also deserves a chance also to in Singapore also I think his fandom is also growing ya I think that's one bes~ the best thing to go and happen in Singapore is that we have\n<Speaker1>: correct correct

        Summary: Speaker1 and Speaker2 discussed their experience and appreciation for Jay-Chou, a singer. Speaker1 mentioned being close to Jay-Chou at an event, almost close enough to touch him, and expressed their admiration for his music. Speaker2 shared that they had already touched Jay-Chou and agreed that he deserves more recognition in Singapore. Both speakers acknowledged Jay-Chou's growing fandom in the country and considered it a positive development.

        Example 2:
        Transcription: <Speaker1>: it allows me to like feel that (um) I shouldn't care what other people think of me I should just do what I want to do how about you have you read any books\n<Speaker2>: ya\n<Speaker1>: what book is the most inspiring to you or what like movie show is the most inspiring to you\n<Speaker2>: no no no no I I read the what the Islamic book\n<Speaker1>: Islamic books\n<Speaker2>: ya Al-Quran\n<Speaker1>: [oh] yes like our religious books\n<Speaker2>: ya (uh) religious books ya\n<Speaker1>: it like it it it makes you it allows you to (um)

        Summary: Speaker1 and Speaker2 discussed the topic of inspiration, with Speaker1 sharing a personal belief that one should not care about others' opinions. Speaker2 mentioned being inspired by a specific book, which is the Islamic holy book, Al-Quran. Both speakers acknowledged the significance of their religious books, with Speaker1 noting that these books can have a profound impact, although the exact nature of this impact was not specified.

        Example 3:
        Transcription: <Speaker1>: my dad using thirty inch not enough for me [ah]\n<Speaker2>: actually why you don't want check in\n<Speaker1>: what check in\n<Speaker2>: you taking Thai Airways cannot check in [meh]\n<Speaker1>: ya ya ya ya ya I'm checking in at cause my mum my mum took one luggage\n<Speaker2>: !huh! why you don't your house no luggage\n<Speaker1>: we only got one thirty inch one one small one large one medium cause we don't travel a lot\n<Speaker2>: but your mum don't travel a lot [meh]\n<Speaker1>: last time she du~ she do [lor]\n<Speaker2>: I got damn lot of luggage at home [one] [leh]\n<Speaker1>: why [ah]\n<Speaker2>: I don't know like buy and then like free\n<Speaker1>: [orh] might might free up the cabin

        Summary: The conversation revolves around luggage and checking in for a flight on Thai Airways. Speaker1 is discussing their family's luggage situation, stating they only have a limited number of suitcases (one 30-inch, one small, one large, and one medium) because they don't travel often. Speaker2 questions the need for more luggage, mentioning that Speaker1's mom also doesn't travel frequently. Speaker1 explains that their mom had previously bought additional luggage. Speaker2 then shares that they have a lot of luggage at home, mostly obtained through purchases or free gifts, and jokingly suggests that Speaker1 might be able to use some of it to free up cabin space.


        You task now.
        Transcription: {context}

        Please output Summary as format below:
        Summary: (Write a concise summary of the dialogue here.)
    """

    transcription=batch["answer"][0]["text"]

    input_message = prompt.format(context = transcription)

    port = random.choice([5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007])
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    models = client.models.list()
    model = models.data[0].id
    chat_response = client.chat.completions.create(
        model      = model,
        messages   = [{"role": "user", "content": input_message},],
        max_tokens = 512,
        n          = 1,
        )
    output=chat_response.choices[0].message.content

    if "Summary:" in output:
        revised_question = random.choice(candidate_instructions)
        revised_answer = output.split('Summary:')[-1].strip()
    else:
        revised_question = random.choice(candidate_instructions)
        revised_answer = "No match found!!!!"

        print("No match found!!!!", flush=True)
        print("output:", output, flush=True)

    batch['instruction'][0]['text']               = revised_question
    batch['answer'][0]['text']                    = revised_answer
    batch['other_attributes'][0]['transcription'] = transcription
    return batch


def build(ROOT_PATH, DATASET_NAME):

    output_path = os.path.join(ROOT_PATH.replace("/ASR", "/DS"), DATASET_NAME.replace("_ASR_", "_DS_"))
    if os.path.exists(output_path):
        print(f"Skipping {output_path} as it already exists")
        return

    data = load_from_disk(os.path.join(ROOT_PATH, DATASET_NAME))    
    data = data.map(map_fn,
                    num_proc          = 224,
                    batched           = True,
                    batch_size        = 1,
                    writer_batch_size = 1,
                    )

    data = data.filter(lambda x: x['instruction']['text'] not in ["No match found!!!!", ""],
                       num_proc          = 224,
                       batch_size        = 1,
                       writer_batch_size = 1,
                       )

    data.save_to_disk(output_path, num_proc=2)



def main(split="all", dataset="all"):
    for ROOT_PATH in ['/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal_bytes/test/ASR',
                      '/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal_bytes/train/ASR']:

        for DATASET_NAME in ['IMDA_PART3_30_ASR_v4',
                             'IMDA_PART4_30_ASR_v4',
                             'IMDA_PART5_30_ASR_v4',
                             'IMDA_PART6_30_ASR_v4_0',
                             'IMDA_PART6_30_ASR_v4_1',
                             'IMDA_PART6_30_ASR_v4_2',
                             'IMDA_PART6_30_ASR_v4_3',
                             'IMDA_PART6_30_ASR_v4_4',
                             'IMDA_PART6_30_ASR_v4_5',
                             'IMDA_PART6_30_ASR_v4_6',
                             'IMDA_PART6_30_ASR_v4_7',
                             'IMDA_PART6_30_ASR_v4_8',
                             'IMDA_PART6_30_ASR_v4_9',

                             'IMDA_PART3_60_ASR_v4',
                             'IMDA_PART4_60_ASR_v4',
                             'IMDA_PART5_60_ASR_v4',
                             'IMDA_PART6_60_ASR_v4',

                             'IMDA_PART3_120_ASR_v4',
                             'IMDA_PART4_120_ASR_v4',
                             'IMDA_PART5_120_ASR_v4',
                             'IMDA_PART6_120_ASR_v4'
                             ]:
            if (split == "all" or split in ROOT_PATH) and (dataset == "all" or dataset in DATASET_NAME):
                build(ROOT_PATH, DATASET_NAME)


if __name__ == "__main__":
    fire.Fire(main)