from huggingface_hub import snapshot_download
import os, sys

print(os.getpid(), flush=True)

# try:
#     folder = snapshot_download(
#         "HuggingFaceFW/fineweb",
#         repo_type="dataset",
#         cache_dir="./cache",
#         local_dir="./fineweb/",
#         resume_download=True,
#         max_workers=8,
#         allow_patterns="data/CC-MAIN-*/*")
#     print(folder, flush=True)
#     print("complete", flush=True)
# except:
#     os.execv(sys.executable, ['python'] + sys.argv)


folder = snapshot_download(
    "HuggingFaceFW/fineweb",
    repo_type="dataset",
    cache_dir="./cache",
    local_dir="./fineweb/",
    resume_download=True,
    max_workers=32,
    allow_patterns="data/CC-MAIN-*/*")
print(folder, flush=True)
print("complete", flush=True)
