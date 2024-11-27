

lines=open('/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4/manifest.jsonl').readlines()

batches = 48

batch_size = len(lines) // batches+1

for i in range(batches):
    with open(f'/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART4/manifest_{i}.jsonl', 'w') as f:
        f.writelines(lines[i*batch_size:(i+1)*batch_size])

