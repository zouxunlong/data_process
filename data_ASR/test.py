
file_path="/scratch/users/astar/ares/suns1/ali_speech_data/r&d/LOTUS-Thai/disk1of2/Supplement/alltext5K.txt"

import chardet
result = chardet.detect(open(file_path, "rb").read())
# f = open(file_path, "r", encoding=result["encoding"]).read()
print(result["encoding"])




# import xml.etree.ElementTree as ET

# # Load the .trs file
# tree = ET.parse("/scratch/users/astar/ares/suns1/ali_speech_data/r&d/SEA speech and annotation 4700161249/Document_Transcription/Tagalog/TG001_TG002/TG001_TG002_Session1c .trs")
# root = tree.getroot()

# # Example: Extracting segments
# for turn in root.findall(".//Turn"):
#     speaker    = turn.get("speaker", "Unknown")
#     start_time = turn.get("startTime")
#     end_time   = turn.get("endTime")
#     print(f"Speaker: {speaker}, Start: {start_time}, End: {end_time}")

#     # Extract words or text within each turn
#     for event in turn.findall("Sync"):
#         print(f"  Sync at {event.get('time')}: {event.tail.strip()}", flush=True)

