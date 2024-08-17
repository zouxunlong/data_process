from concurrent.futures import ThreadPoolExecutor
import os
from utils import remove_empty, download



def main(
        output_dir: str = "./youtube8m",
        id_files_share: list[str] = [],
        error_ids: list[str] = [],
):
    while id_files_share:

        id_file = id_files_share.pop()
        directory = os.path.join(output_dir, id_file.split('/')[-1].split(".")[0])

        print("tasks for id_file: {} started.".format(id_file), flush=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, id in enumerate(open(id_file).readlines()):
                id = id.strip()

                if id in ["AccessDenie"] + error_ids:
                    continue

                filename = '{}.wav'.format(id)
                if os.path.exists(os.path.join(directory, filename)):
                    continue

                pool.submit(download, id, directory, filename)
            print("tasks for id_file: {} submited.".format(id_file), flush=True)
        
        print("tasks for id_file: {} completed.".format(id_file), flush=True)


if __name__ == "__main__":

    remove_empty('./youtube8m')

    print("main process id {} starts.".format(os.getpid()), flush=True)
    open("./pid.log", "w", encoding="utf8").write(str(os.getpid())+" ")
    output_dir = "./youtube8m/08"
    id_file_dir = "./category-ids/08"
    id_files = os.listdir(id_file_dir)
    id_files.sort()
    id_files = [os.path.join(id_file_dir, id_file) for id_file in id_files]
    error_ids = [line.split()[0] for line in open("./error_ids.log").readlines()]

    main(output_dir=output_dir,
         id_files_share=id_files,
         error_ids=error_ids
         )

    print("all finished.", flush=True)
