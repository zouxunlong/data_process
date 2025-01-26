


for part in ["PART3", "PART4", "PART5", "PART6"]:

    lines=open(f'/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest.jsonl').readlines()
    batches = 80
    batch_size = len(lines) // batches + 1
    for i in range(batches):
        with open(f'/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest_{i}.jsonl', 'w') as f:
            f.writelines(lines[i*batch_size:(i+1)*batch_size])

