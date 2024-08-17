from datasets import load_dataset
import os

# for lang in ['ar', 'arz', 'as', 'ast', 'av', 'az', 'azb', 'ba', 'bar', 'bcl', 'be', 'bg', 'bh', 'bn', 'bo', 'bpy', 'br', 'bs', 'bxr', 'ca', 'cbk', 'ce', 'ceb', 'ckb', 'cs', 'cv', 'cy', 'da', 'de', 'dsb', 'dv', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'frr', 'fy', 'ga', 'gd', 'gl', 'gn', 'gom', 'gu', 'he', 'hi', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'ie', 'ilo', 'io', 'is', 'it', 'ja', 'jbo', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'krc', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lez', 'li', 'lmo', 'lo', 'lrc', 'lt', 'lv', 'mai', 'mg', 'mhr', 'min', 'mk', 'ml', 'mn', 'mr', 'mrj', 'mt', 'mwl', 'my', 'myv', 'mzn', 'nah', 'nap', 'nds', 'ne', 'new', 'nl', 'nn', 'no', 'oc', 'or', 'os', 'pa', 'pam', 'pl', 'pms', 'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'rue', 'sa', 'sah', 'scn', 'sd', 'sh', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'tyv', 'ug', 'uk', 'ur', 'uz', 'vec', 'vi', 'vls', 'vo', 'wa', 'war', 'wuu', 'xal', 'xmf', 'yi', 'yo', 'yue', 'zh']:
def download(lang):
    dataset = load_dataset("uonlp/CulturaX", lang)
    dataset.save_to_disk("/mnt/home/zoux/datasets/xunlong_working_repo/datasets_text/data_text/data_CulturaX/CulturaX.{}".format(lang))
    print("complet {}".format(lang), flush=True)

if __name__=="__main__":
    print(os.getpid(), flush=True)
    for lang in ['zh']:
        download(lang)
    print("complet all", flush=True)