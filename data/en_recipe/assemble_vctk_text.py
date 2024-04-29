import os
import phonemizer
import sys
sys.path.append(os.getcwd())
import argparse
import glob

from tqdm import tqdm
def main(args):
    root = args.root
    full_list = []
    speaker_sub_dirs = glob.glob(os.path.join(root, "wav48", "p*"))
    for subdir in tqdm(speaker_sub_dirs):
        wavfiles = glob.glob(os.path.join(subdir, "*.wav"))
        txtfiles = [f.replace("wav48", "transcripts").replace(".wav", ".txt") for f in wavfiles]
        for txt in txtfiles:
            if not os.path.exists(txt):
                continue
            with open(txt, "r") as f:
                text = f.readlines()[0]
                text = text.replace("\t", "")
                sid = subdir.split("\\")[-1]
                wav = txt.replace("transcripts", "wav48").replace(".txt", ".wav")
                line = f"{wav}\t{sid}\t{text}"
                full_list.append(line)

    processed_text = []
    for line in tqdm(full_list):
        wav_path, sid, text = line.split("\t")
        text_path = wav_path.replace("wav48", "texts").replace(".wav", ".txt")
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                processed_text.append(f.readlines()[0].rstrip("\n"))
        else:
            print(f"Missing: {text_path}")

    with open(os.path.join(root, "clean_transcripts.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(processed_text))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='E:/datasets/VCTK/')
    args = parser.parse_args()

    main(args)