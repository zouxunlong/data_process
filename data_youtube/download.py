
import os
from utils import remove_empty, download


def main(id_file, directory, error_ids):

        for i, id in enumerate(open(id_file).readlines()):
            id = id.strip()
            if id in ["AccessDenie"]+error_ids:
                continue

            filename = '{}.wav'.format(id)
            if os.path.exists(os.path.join(directory, filename)):
                continue

            download(id, directory, filename)


if __name__ == "__main__":

    remove_empty('./youtube8m')
    
    id_file = "./category-ids/06/06_dn.txt"
    directory = "./youtube8m/06/06_dn"
    error_ids = [line.split()[0] for line in open("./error_ids.log").readlines()]

    main(id_file, directory, error_ids)