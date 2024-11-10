from datasets import load_from_disk

ds=load_from_disk("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/datasets_multimodal/test/DS/IMDA_PART3_30_DS_v4")

print(ds[0])
print(ds[10])
print(ds[220])
print(ds[440])
print(ds[980])
