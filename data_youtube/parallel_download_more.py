from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Manager
import os
import time
from func_timeout import FunctionTimedOut, func_set_timeout
from pytube import YouTube
import psutil



@func_set_timeout(60)
def download_video(id, directory, filename, error_file):
    try:
        video_url = 'https://www.youtube.com/watch?v={}'.format(id)
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()
        stream.download(directory, filename=filename)
        print("DOWNLOADED: {} at {}".format(id, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))), flush=True)
    except Exception as e:
        if "Forbidden" in str(e):
            os.remove(os.path.join(directory, filename))
            print("{} ERROR: {}".format(id, e), flush=True)
        else:
            print("{} ERROR: {}".format(id, e), flush=True)
            open(error_file, "a", encoding="utf8").write("{} ERROR: {}".format(id, e) + "\n")


def download(id, directory, filename, error_file):
    try:
        download_video(id, directory, filename, error_file)
    except FunctionTimedOut as e:
        print("{} ERROR: {}".format(id, e.msg.strip()), flush=True)
        open("./jumpped_ids.log", "a", encoding="utf8").write("{} ERROR: {}".format(id, e.msg.strip()) + "\n")



def main(
        output_dir: str = "./audios_more",
        id_files_share: list[str] = [],
):
    while len(id_files_share)>0:

        id_file = id_files_share.pop()
        error_file=id_file.replace(".txt", ".errors")
        directory = os.path.join(output_dir, id_file.split("/")[-1].split(".")[0])
        if os.path.exists(error_file):
            error_ids = [line.split()[0] for line in open(error_file).readlines()]
        else:
            error_ids=[]

        print("tasks for id_file: {} started.".format(id_file), flush=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, id in enumerate(open(id_file).readlines()):
                id = id.strip()

                if id in error_ids:
                    continue

                filename = '{}.wav'.format(id)
                if os.path.exists(os.path.join(directory, filename)):
                    continue

                pool.submit(download, id, directory, filename, error_file)
            print("tasks for id_file: {} submited.".format(id_file), flush=True)
        
        print("tasks for id_file: {} completed.".format(id_file), flush=True)


if __name__ == "__main__":

    open("./pid.log", "w", encoding="utf8").write(str(os.getpid())+" ")
    print("main process id {} starts.".format(os.getpid()), flush=True)

    output_dir = "./audios_more"
    id_file_dirs = ["./ids_more"]

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

    for i in list(range(4)):
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
