from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Manager
import psutil
import os
from utils import download


def main(
        output_dir: str = "./y8m_audios",
        id_files_share: list[str] = [],
):
    while len(id_files_share)>0:

        id_file = id_files_share.pop()
        error_file=id_file.replace(".txt", ".errors")
        directory = output_dir
        if os.path.exists(error_file):
            error_ids = [line.split()[0] for line in open(error_file).readlines()]
        else:
            error_ids=[]

        print("tasks for id_file: {} started.".format(id_file), flush=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, id in enumerate(open(id_file).readlines()):
                id = id.strip()

                if id in ["AccessDenie"] + error_ids:
                    continue

                filename = '{}.wav'.format(id)

                pool.submit(download, id, directory, filename, error_file)
            print("tasks for id_file: {} submited.".format(id_file), flush=True)
        
        print("tasks for id_file: {} completed.".format(id_file), flush=True)


if __name__ == "__main__":

    open("./jumpped_ids.log", "a", encoding="utf8").write("")
    open("./pid.log", "w", encoding="utf8").write(str(os.getpid())+" ")
    print("main process id {} starts.".format(os.getpid()), flush=True)

    output_dir = "./y8m_audios"
    id_file_dirs = ["./ids_category_rest"]

    id_files=[]
    for id_file_dir in id_file_dirs:
        for parent_dir, dirs, files in os.walk(id_file_dir):
            for file in files:
                if file.endswith(".txt"):
                    id_files.append(os.path.join(parent_dir, file))
    id_files.sort()

    manager = Manager()
    id_files_share = manager.list()
    id_files_share.extend(id_files)

    process_list = []

    for i in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]:
        process = Process(target=main, args=(output_dir, id_files_share))
        process.start()
        process_list.append(process)
        p = psutil.Process(process.pid)
        p.cpu_affinity([i])
        print("process id {} starts on cpu {}".format(process.pid, i), flush=True)
        open("./pid.log", "a", encoding="utf8").write(str(process.pid)+" ")

    for res in process_list:
        res.join()
    print("processes all finished.", flush=True)
