import os
import shutil

types=set()
for parent_dir, dirs, files in os.walk("/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/for_extraction/origin"):
    for file in files:
        type=os.path.basename(file).split(".")[-1]
        shutil.copyfile(os.path.join(parent_dir, file), "/mnt/data/all_datasets/xunlong_working_repo/_data_in_processing/mt_data/for_extraction/categorized/"+type+"/"+file)    

