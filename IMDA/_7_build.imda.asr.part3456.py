import os
from datasets import load_from_disk, concatenate_datasets
import fire
import shutil
from glob import glob

def filt_fn_test(batch, part, split, test_conversation_ids):

    if part in ["PART3", "PART4"]:
        results=[item["conversation_id"] in test_conversation_ids for item in batch["other_attributes"]]
    if part in ["PART5"]:
        results=[item["conversation_id"]+"_"+item["setting"] in test_conversation_ids for item in batch["other_attributes"]]
    if part in ["PART6"]:
        results=[item["conversation_id"]+"_cc-"+item["setting"] in test_conversation_ids for item in batch["other_attributes"]]    

    if split == "test":
        return results
    if split == "train":
        return [not i for i in results]


def seperate(split_path):

    part = split_path.split("/")[-1].split("_")[1]
    if part == "PART3":
        test_conversation_ids={'2726', '3260', '3252', '2642', '3068', '3093', '3037', '3107', '3138'}
    if part == "PART4":
        test_conversation_ids={'0159', '0816', '1239', '0088', '2020', '0369', '0903', '0461', '0464', '0211', '0680', '0941', '0952', '0485', '0301'}
    if part == "PART5":
        test_conversation_ids={'3367_fin', '3447_pos', '3450_neg', '3250_pos', '4156_deb-1', '3100_pos', '3232_fin', '4198_deb-3', '3128_neg', '3177_neg', '3294_neg', '3313_neg', '3348_neg', '4137_deb-2', '3209_neg', '3459_pos', '3186_fin', '3504_fin', '3762_pos', '4219_deb-3', '3540_fin', '3388_pos', '4168_deb-2', '4128_deb-2', '3104_neg', '3548_pos', '4041_deb-3', '3759_fin', '3004_fin', '4026_deb-2', '4082_deb-1', '3237_pos', '3083_pos', '4067_deb-3', '3433_pos', '3226_pos', '4242_deb-1', '4224_deb-2', '3095_pos', '3277_pos', '4187_deb-2', '4079_deb-2', '3723_pos', '3370_pos', '4069_deb-3', '3606_neg', '3178_neg', '3733_fin', '3773_pos', '3523_fin', '3642_fin', '3718_pos', '3199_pos', '3068_neg', '3695_pos', '3416_pos', '3321_fin', '4091_deb-3', '3229_fin'}
    if part == "PART6":
        test_conversation_ids={'0369_cc-hol', '1130_cc-bnk', '0302_cc-hol', '0247_cc-res', '1340_cc-bnk', '1436_cc-hdb', '1803_cc-hdb', '0266_cc-res', '0106_cc-hol', '0169_cc-hol', '0392_cc-res', '1123_cc-ins', '0488_cc-res', '0893_cc-tel', '1286_cc-bnk', '1816_cc-hdb', '0365_cc-res', '1968_cc-msf', '1831_cc-moe', '0927_cc-bnk', '1751_cc-msf', '0144_cc-hol', '1000_cc-ins', '0508_cc-res', '1342_cc-tel', '1776_cc-moe', '1292_cc-ins', '1692_cc-hdb', '1989_cc-moe', '1371_cc-msf', '0879_cc-ins', '1188_cc-ins', '0439_cc-hot', '0297_cc-hot', '0877_cc-bnk', '0550_cc-hot', '1422_cc-moe', '0573_cc-hot', '0681_cc-hol', '1308_cc-bnk', '1911_cc-hdb', '1960_cc-hdb', '0152_cc-hol', '1535_cc-moe', '0801_cc-ins', '0942_cc-bnk', '1546_cc-moe', '1336_cc-bnk', '1447_cc-hdb', '0177_cc-hot', '0129_cc-res', '1481_cc-moe', '1316_cc-tel', '0136_cc-hol', '1152_cc-bnk', '1071_cc-ins', '0420_cc-res', '1949_cc-moe', '0627_cc-hol', '1982_cc-msf', '1745_cc-hdb', '1267_cc-tel', '1644_cc-moe', '0620_cc-hot', '1626_cc-msf', '0983_cc-ins', '1522_cc-moe', '1496_cc-msf', '0049_cc-hol', '0304_cc-hol', '1882_cc-moe', '0973_cc-ins', '1584_cc-msf', '1542_cc-moe', '0269_cc-hot', '1164_cc-ins', '1283_cc-bnk', '0244_cc-hot', '0970_cc-bnk', '0751_cc-ins', '0172_cc-res', '0268_cc-hot', '0069_cc-res', '1670_cc-msf', '1568_cc-moe', '1466_cc-msf', '1051_cc-ins', '1684_cc-hdb', '1663_cc-moe', '0728_cc-tel', '1903_cc-moe', '1441_cc-msf', '1892_cc-msf', '1028_cc-tel', '1957_cc-hdb', '0416_cc-res', '0920_cc-tel', '0409_cc-res', '1966_cc-hdb', '2016_cc-moe', '0700_cc-ins', '1528_cc-msf', '0649_cc-hol', '0944_cc-bnk', '0215_cc-hot', '2032_cc-hdb', '0708_cc-bnk', '1756_cc-hdb', '1252_cc-ins', '0572_cc-hol', '1975_cc-hdb', '0610_cc-hot', '1687_cc-hdb', '0074_cc-hol', '0454_cc-hol', '0858_cc-bnk', '1460_cc-hdb', '0643_cc-hol', '1759_cc-hdb'}

    ds = load_from_disk(split_path)

    ds_test  = ds.filter(filt_fn_test, 
                         fn_kwargs={"part":part, 
                                    "split":"test",
                                    "test_conversation_ids": test_conversation_ids },
                         batched=True, 
                         num_proc=112)
    print("ds_test: ", len(ds_test), flush=True)
    ds_test.save_to_disk(split_path.replace("_v5", "").replace("/train/", "/test/"), num_proc=10)

    ds_train = ds.filter(filt_fn_test, 
                         fn_kwargs={"part":part, 
                                    "split":"train",
                                    "test_conversation_ids": test_conversation_ids },
                         batched=True, 
                         num_proc=112)
    print("ds_train: ", len(ds_train), flush=True)
    ds_train.save_to_disk(split_path.replace("_v5", ""), num_proc=10)


def main():
    dirs=glob("/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/train/*/*_v5")
    for dir in sorted(dirs):
        print("start {} ".format(dir), flush=True)
        seperate(split_path=dir)
    print("complete all", flush=True)


def filter_deb():
    splits=[

        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_30_AR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_30_GR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_30_MIX_v4",

        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_60_AR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_60_GR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_60_MIX_v4",
        
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_120_AR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_120_GR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_120_MIX_v4",

        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_300_AR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_300_GR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_300_MIX_v4",

        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_AR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_GR_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA/IMDA_PART5_MIX_v4",

        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SDS/IMDA_PART5_30_SDS_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SDS/IMDA_PART5_60_SDS_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SDS/IMDA_PART5_120_SDS_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SDS/IMDA_PART5_300_SDS_v4",

        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SQA/IMDA_PART5_30_SQA_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SQA/IMDA_PART5_60_SQA_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SQA/IMDA_PART5_120_SQA_v4",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SQA/IMDA_PART5_300_SQA_v4",
    ]
    for split_path in splits:
        print("start {} ".format(split_path), flush=True)
        ds=load_from_disk(split_path)
        new_ds=ds.filter(lambda batch: [x["setting"] !="deb" for x in batch["other_attributes"]], batched=True, num_proc=64)
        if len(new_ds)!=len(ds):
            new_ds.save_to_disk(split_path.replace("_v4", "_v5"), num_proc=1)
            shutil.rmtree(split_path)
            os.rename(split_path.replace("_v4", "_v5"), split_path)
    print("complete all", flush=True)


def combine():
    
    def map_fn(batch):
        for other_attributes in batch["other_attributes"]:
            other_attributes.setdefault("need_further_proecess", False)
            other_attributes.setdefault("transcription", "")
        return batch

    dirs=[
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/PQA",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SDS",
        "/data/projects/13003558/zoux/workspaces/data_process/_data_in_processing/imda/imda_bytes/test/SQA",
    ]
    for dir in dirs:
        splits=os.listdir(dir)
        for split in splits:
            if "PART1" in split or "PART2" in split:
                continue

            split_test=os.path.join(dir, split)
            split_train=split_test.replace("/test/", "/train/")
            if os.path.exists(split_train.replace("_v4", "_v5")):
                continue

            print("start {} ".format(split_test), flush=True)
            ds_test=load_from_disk(split_test)
            ds_train=load_from_disk(split_train)

            ds_test=ds_test.map(map_fn, 
                                batched=True, 
                                batch_size=100, 
                                writer_batch_size=1, 
                                features=ds_train.features,
                                num_proc=10)

            print(ds_test.features["other_attributes"].keys(), flush=True)
            print(ds_train.features["other_attributes"].keys(), flush=True)
            ds=concatenate_datasets([ds_train, ds_test])
            ds.save_to_disk(split_train.replace("_v4", "_v5"), num_proc=10)
            # shutil.rmtree(split_train)
            # os.rename(split_train.replace("_v4", "_v5"), split_train)
    print("complete all", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
