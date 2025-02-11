


for part in ["PART3", "PART4", "PART5", "PART6"]:

    lines=open(f'/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest.jsonl').readlines()
    batches = 8
    batch_size = len(lines) // batches

    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest0.jsonl", "w", encoding="utf-8").writelines(lines[:batch_size])
    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest1.jsonl", "w", encoding="utf-8").writelines(lines[batch_size:2*batch_size])
    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest2.jsonl", "w", encoding="utf-8").writelines(lines[2*batch_size:3*batch_size])
    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest3.jsonl", "w", encoding="utf-8").writelines(lines[3*batch_size:4*batch_size])
    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest4.jsonl", "w", encoding="utf-8").writelines(lines[4*batch_size:5*batch_size])
    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest5.jsonl", "w", encoding="utf-8").writelines(lines[5*batch_size:6*batch_size])
    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest6.jsonl", "w", encoding="utf-8").writelines(lines[6*batch_size:7*batch_size])
    open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest7.jsonl", "w", encoding="utf-8").writelines(lines[7*batch_size:])
