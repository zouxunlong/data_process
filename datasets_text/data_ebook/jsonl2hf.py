from datasets import load_dataset


ds=load_dataset("json", data_files="/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_ebook/book.archive.id.jsonl")
ds.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_ebook/book_archive_id", num_proc=10)
print("complete archive id", flush=True)

ds=load_dataset("json", data_files="/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_ebook/book.archive.ms.jsonl")
ds.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_ebook/book_archive_ms", num_proc=10)
print("complete archive ms", flush=True)

ds=load_dataset("json", data_files="/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_ebook/book.google.id.jsonl")
ds.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_ebook/book_google_id", num_proc=10)
print("complete google id", flush=True)

