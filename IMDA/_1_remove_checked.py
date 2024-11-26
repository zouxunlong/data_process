
from glob import glob
import os

error_files=os.listdir("/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6/_error")


for file in error_files:
    if file.endswith(".wav"):
        try:
            os.remove(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6/Audio/{file}")
        except FileNotFoundError:
            pass
    if file.endswith(".TextGrid"):
        try:
            os.remove(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/PART6/Scripts/{file}")
        except FileNotFoundError:
            pass
