import os
os.environ['TOKENIZERS_PARALLELISM'] = "true"
import sys
sys.path.append(os.getcwd())
import argparse
import random
from tqdm import tqdm
import torchaudio
from utils.g2p.english import g2p
import soundfile as sf
from multiprocessing import Pool, Manager, cpu_count
import glob

ROOT = ""
SPLIT = ""
full_list = []

def check_progress():
    global ROOT, SPLIT, full_list
    root = ROOT
    split = SPLIT
    n_exist = 0
    for line in full_list:
        wav_path, sid, text = line.split("\t")
        text_path = wav_path.replace("wav48", "texts").replace(".wav", ".txt")
        if os.path.exists(text_path):
            n_exist += 1

    print(f"Progress: {n_exist}/{len(full_list)}")

def process_line(line):
    global ROOT, SPLIT
    root = ROOT
    split = SPLIT
    language = "en-us"
    wav_path, sid, text = line.split("\t")
    txt_path = wav_path.replace("wav48", "texts").replace(".wav", ".txt")
    # if os.path.exists(txt_path):
    #     return None
    # assert os.path.exists(wav_path)
    formatted_sid = f"vctk-{language}-{sid}"

    # phonemize
    text_norm = text.rstrip("\n")
    phones = g2p(text_norm, language=language,)

    # check audio length
    if os.path.exists(wav_path):
        wav, sr = sf.read(wav_path)
        if len(wav) < sr * 1 or len(wav) > sr * 60:
            del wav, sr
            return None
        del wav, sr
    else:
        return None

    if not len(text) or len(text) > 500:
        return None
        # continue

    # assert len(text_cleaner(sep_phones)) == len(sep_phones)

    processed_line = f"{wav_path}\t{formatted_sid}\t{language}\t{text_norm}\t{phones}\n"
    # write the line to txt file in another folder
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(processed_line)

    if random.random() < 1e-2:
        # check_progress()
        print(f"Processed {txt_path}")

def process_line_wrapper(line):
    result = process_line(line)
def main(args):
    root = args.root
    global ROOT, full_list
    ROOT = root
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


    num_processes = args.num_processes
    if num_processes == 1:
        for line in tqdm(full_list):
            process_line(line)
    else:
        print(f"Using {num_processes} processes")
        with Pool(processes=num_processes) as pool:
            pool.map(process_line_wrapper, full_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/azureuser/vctk/')
    parser.add_argument('--num_processes', type=int, default=cpu_count())
    args = parser.parse_args()

    main(args)