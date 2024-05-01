import soundfile as sf
import os
import random
from tqdm import tqdm

prepared_transcripts = [
    "/home/azureuser/vctk/clean_transcripts.txt",
]

target_dir = "./data/"

def main():
    clean_list = []
    raw_list = []

    for transcript in prepared_transcripts:
        with open(transcript, "r", encoding="utf-8") as f:
            raw_list += f.readlines()

    for line in tqdm(raw_list):
        wav_path, sid, language, text_norm, phones = line.split("\t")
        if not os.path.exists(wav_path):
            continue
        wav, sr = sf.read(wav_path)
        if len(wav) < sr * 1 or len(wav) > sr * 60:
            continue
        clean_list.append(line)

    # split train val
    random.shuffle(clean_list)
    train_list = clean_list[:int(len(clean_list) * 0.9)]
    val_list = clean_list[int(len(clean_list) * 0.9):]

    with open(os.path.join(target_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.writelines(train_list)
    with open(os.path.join(target_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.writelines(val_list)

if __name__ == '__main__':
    main()