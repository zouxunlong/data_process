import os
import time
from func_timeout import FunctionTimedOut, func_set_timeout
from pytube import YouTube


def remove_empty(dir):
    print("remove empty starts.", flush=True)
    for parent_dir, dirs, files in os.walk(dir):
        n=0
        dirs.sort()
        files.sort()
        for file in files:
            if os.path.getsize(os.path.join(parent_dir, file)) ==0:
                os.remove(os.path.join(parent_dir, file))
                n+=1
                continue
            if not file.endswith(".wav"):
                os.remove(os.path.join(parent_dir, file))
                n+=1
                continue
        print("complete {}, removed {}".format(parent_dir, n), flush=True)
    print("remove empty all completes.", flush=True)


def remove_forbidden(dir):
    print("remove forbidden starts.", flush=True)
    for parent_dir, dirs, files in os.walk(dir):
        n=0
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith(".errors"):
                lines=open(os.path.join(parent_dir, file)).readlines()
                lines=[line for line in lines if "Forbidden" not in line]
                open(os.path.join(parent_dir, file), "w").write("".join(lines))
                print("complete {}".format(os.path.join(parent_dir, file)), flush=True)
    print("remove forbidden all completes.", flush=True)


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


def count(dir):
    count_error=0
    count_ids=[]
    for parent_dir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".errors"):
                path=os.path.join(parent_dir, file)
                count_error+=len(open(path).readlines())
            if file.endswith(".txt"):
                path=os.path.join(parent_dir, file)
                count_ids+=open(path).readlines()
    print("\n{}:\n count_ids: {},\n count_error: {},\n should get: {}\n".format(dir, len(count_ids), count_error, len(count_ids)-count_error), flush=True)

    unique_ids=set(count_ids)
    print("unique_ids: {}".format(len(unique_ids)), flush=True)


def category_ids(dir):
    count_ids=[]
    for parent_dir, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".txt"):
                path=os.path.join(parent_dir, file)
                count_ids+=open(path).readlines()

    unique_ids=set(count_ids)
    print("unique_ids: {}".format(len(unique_ids)), flush=True)
    open("category_ids.txt", "w", encoding="utf8").write("".join(unique_ids))

def more_ids():
    video_ids=set(open("video_ids.txt").readlines())
    category_ids=set(open("category_ids.txt").readlines())
    print("video_ids: {}".format(len(video_ids)), flush=True)
    print("category_ids: {}".format(len(category_ids)), flush=True)
    more_ids=video_ids-category_ids
    print("more_ids: {}".format(len(more_ids)), flush=True)
    open("more_ids.txt", "w", encoding="utf8").write("".join(more_ids))

def split():
    ids=open("ids_more.txt").readlines()
    total=len(ids)
    chunk=total//60
    for i in range(61):
        open("ids_more/{}.txt".format(i), "w", encoding="utf8").write("".join(ids[chunk*i:chunk*(i+1)]))

if __name__=="__main__":
    dirs=["audios_more"]
    for dir in dirs:
        for parent_dir, dirs, files in os.walk(dir):
            dirs.sort()
            files.sort()
            for file in files:
                sub_dir=file[:2]
                os.makedirs("./y8m/{}".format(sub_dir), exist_ok=True)
                os.rename("./{}/{}".format(parent_dir, file), "./y8m/{}/{}".format(sub_dir, file))
        print("complete {}".format(parent_dir))

