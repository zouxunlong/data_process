import json
import random
from tqdm import tqdm
from datasets import load_from_disk


def get_dialogues():
    lines = []
    for file in ["dream-test.jsonl", "dream-train.jsonl", "dream-validation.jsonl"]:
        lines.extend(open(file=file).readlines())

    data = [json.loads(line) for line in lines]

    dialogues = {}

    for sample in tqdm(data):
        dialogue_id = sample['dialogue_id']
        if dialogue_id in dialogues.keys():
            continue
        dialogue = sample["dialogue"]
        dialogues[dialogue_id] = dialogue

    with open("dialogues.jsonl", "w", encoding="utf-8") as f:
        for dialogue_id, dialogue in dialogues.items():
            f.write(json.dumps(
                {"dialogue_id": dialogue_id, "dialogue": dialogue})+"\n")


def extract_titles():
    titles = {}
    with open("dialogues_title_utterance.jsonl") as f, \
            open("dialogues_title_utterance_new.jsonl", "w", encoding="utf-8") as f_out:
        for line in f:
            item = json.loads(line)
            dialogue = item["dialogue"]
            titles = []
            for sentence in dialogue:
                title = sentence[0]
                discource = sentence[1]
                if title not in titles:
                    titles.append(title)
            item["titles"] = titles
            f_out.write(json.dumps(item)+"\n")


def get_all_titles():
    titles_set = set()
    with open("dialogues_new.jsonl") as f:
        for line in f:
            item = json.loads(line)
            titles = item["titles"]
            for title in titles:
                titles_set.add(title)
    print(titles_set)


def split_title_utterance():
    titles_set = set()
    with open("dialogues_new.jsonl") as f, \
            open("dialogues_title_utterance.jsonl", "w", encoding="utf-8") as f_out:
        for line in f:
            item = json.loads(line)
            dialogue = [tuple(sentence.split(":", 1))
                        for sentence in item["dialogue"]]
            item["dialogue"] = dialogue
            del item["titles"]
            f_out.write(json.dumps(item)+"\n")

    print(titles_set)


def title_to_speaker():

    males = ['Ethan', 'Chef Randall', 'Mr. Taylor', 'Bill', 'Berry', 'Andrew', 'Ed', 'Boy', 'Li Ming', 'John Knox', 'Michael', 'Gavin', 'M', 'Mr. Yuan', 'Husband', 'Charles', 'Caller', 'Frank', 'Driving Officer', 'Tod', 'Passenger', 'Micky', 'Tutor', 'K', 'Mad', 'Dan', 'Rental Car Agent', 'Dentist', 'Mr. Dong', 'Heather', 'Presenter', 'Mathew', 'News Reporter', 'Receptionist', 'Henry', 'Robber', 'Man', 'Shawn', 'Car Owner', 'Rocky', 'Patient', 'Mr. Adams', 'Hank', 'Stuart', 'Phil', 'Program Host', 'James', 'Young Man', 'Waiter', 'Brandon', 'Roger', 'Randall', 'Jake',
             'Son', 'Taxi Driver', 'Apartment Manager', 'John',  'Host', 'Police Officer', 'Doug', 'Steve', 'Justin', 'Merchant', 'Jason', 'Tim', 'Jack', 'Crystal', 'Daniel', 'Delia Robinson', 'B', 'Driver', 'Apartment Owner', 'Norman', 'Carl', 'Student', 'Scott', 'Dad', 'Tenant', 'Pete', 'Kids', 'Jacob', 'Paul', 'Fisher', 'Pancho', 'Dave', 'Teacher', 'Tom', 'Markus', 'Ryan', 'Mark', 'Customer', 'Ron', 'David', 'Father', 'Car Salesman', 'Nick', 'Terry', 'Mechanic', 'Ted', 'Guest', 'Dean', 'Joshua', 'Ronald', 'Customs Officer', 'Peter', 'Mr. Smith', 'Greg', 'Sam', 'Bank Teller']
    females = ['Mrs. Smith', "Andrew's Sister", 'Sarah', 'Neighbor', 'Game Show Host', 'Jane', 'Stacy', 'F', 'Susan', 'Sales Associate', 'Jenny', 'Maria', 'Ann', 'Carla', 'W', 'Daughter', 'Julie', 'Server', "Dave's Sister", 'Jan', 'Store Employee', 'Kelly', 'Mother', 'Mary', 'Security', 'Brenda', 'Amanda', 'Ashley', 'Interviewer', 'Ranae',
               'Stephanie', 'Wife', 'Nate', 'Hotel Clerk', 'Little Girl', 'Beautician', 'Josh', 'Employee', 'Florist', 'Lisa', 'Mum', 'Alex', 'Travel Agent', 'A', 'Operator', 'W1', 'Woman', 'Girl', 'Amy', 'Older Sister', 'Rachel', 'Vet', 'Emily', 'Diane', 'Jori', 'Telemarketer', 'Sara', 'Nancy', 'Ross', 'The Big Sister', 'R', 'Parent', 'W2', 'Anna Maria']

    male_speakers = ['Aaron Dreschner', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Damjan Chapman', 'Dionisio Schuyler', 'Gilberto Mathias',
                     'Ilkin Urbano', 'Kazuhiko Atallah', 'Kumar Dahl', 'Ludvig Milivoj', 'Luis Moray', 'Marcos Rudaski', 'Royston Min', 'Torcull Diarmuid', 'Viktor Eka', 'Viktor Menelaos', 'Wulf Carlevaro']
    female_speakers = ['Alexandra Hisakawa', 'Alison Dietlinde', 'Ana Florence', 'Andrew Chipper', 'Annmarie Nele', 'Asya Anara', 'Badr Odhiambo', 'Barbora MacLean', 'Brenda Stern', 'Chandra MacFarland', 'Claribel Dervla',
                       'Daisy Studious', 'Gitta Nikolina', 'Henriette Usha', 'Lilya Stainthorpe', 'Maja Ruoho', 'Rosemary Okafor', 'Sofia Hellen', 'Suad Qasim', 'Tammie Ema', 'Tammy Grit', 'Tanja Adelina', 'Uta Obando', 'Vjollca Johnnie',]

    with open("dialogues_title_utterance_new.jsonl") as f, \
            open("dialogues_title_utterance_new_speaker.jsonl", "w", encoding="utf-8") as f_out:
        for line in f:
            item = json.loads(line)
            titles = item["titles"]
            title_to_speaker = {}

            for title in titles:
                if title in males:
                    title_to_speaker[title] = random.choice(male_speakers)
                elif title in females:
                    title_to_speaker[title] = random.choice(female_speakers)
                else:
                    print("error")

            item["title_to_speaker"] = title_to_speaker
            if not len(titles) == 2:
                print("special")
            f_out.write(json.dumps(item)+"\n")


def final():

    with open("dialogues_title_utterance_new_speaker.jsonl") as f, \
            open("dialogues_final.jsonl", "w", encoding="utf-8") as f_out:
        for line in f:
            item = json.loads(line)
            dialogue = item["dialogue"]
            title_to_speaker = item["title_to_speaker"]

            new_dialogue = []

            for title, utterance in dialogue:
                speaker = title_to_speaker[title]
                new_dialogue.append([speaker, utterance])

            item["dialogue"] = new_dialogue

            del item["title_to_speaker"]
            del item["titles"]

            f_out.write(json.dumps(item)+"\n")


def refine_dream():
    dialogue_ids = set()
    with open("dialogues_final.jsonl") as f:
        for line in f:
            item = json.loads(line)
            dialogue_ids.add(item["dialogue_id"])
    with open("dream-validation.jsonl") as dream_test, \
            open("dream-validation.filtered.jsonl", "w", encoding="utf-8") as refined_dream_test:
        for line in dream_test:
            item = json.loads(line)
            if item["dialogue_id"] in dialogue_ids:
                refined_dream_test.write(json.dumps(item)+"\n")


def reformat_dream():

    with open("dream-test.filtered.jsonl") as dream, \
            open("dream-test.filtered.reformated.jsonl", "w", encoding="utf-8") as dream_reformated:
        for line in dream:
            item = json.loads(line)
            choices = item["choice"]
            answer = item["answer"]
            choice_indexs = ["(E)", "(D)", "(C)", "(B)", "(A)"]
            new_choices = []
            for choice in choices:
                choice_index = choice_indexs.pop()
                new_choice = choice_index+" " + choice
                new_choices.append(new_choice)
                if choice == answer:
                    answer = new_choice
            item["choice"] = new_choices
            item["answer"] = answer

            dream_reformated.write(json.dumps(item)+"\n")


def examine_format(file):
    answer_inits=set()
    with open(file) as f:
        for line in f:
            item=json.loads(line)
            answer=item["answer"]
            answer_inits.add(answer[:4])
    print(answer_inits,flush=True)


def build_hf_dataset():
    with open("dream-train.filtered.reformated.jsonl") as f,\
        open("dream.train.jsonl", "w", encoding="utf-8") as dream_reformated:
        for line in f:
            item=json.loads(line)
            item["audio"]="/home/xuanlong/data_prepare_aqa/tts_audio2/{}.wav".format(item["dialogue_id"])
            dream_reformated.write(json.dumps(item)+"\n")

def filter_short(example):
    length = len(example["audio"]["array"])
    if length < 72000:
        return False
    return True


def filter_none(example):
    array=example["audio"]["array"]
    if array.any():
        return True
    else:
        return False

if __name__ == "__main__":

    dataset_path = "/home/user/data/data_SQA/dream_v1.schemed/validation"
    dataset = load_from_disk(dataset_path)
    # updated_dataset = dataset.filter(filter_short)
    # updated_dataset.save_to_disk("{}_filtered".format(dataset_path))
    print(dataset[34], flush=True)

