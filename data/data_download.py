# data/data_download.py

import deepchem as dc
import pandas as pd
import os
import requests
import gzip
import shutil

def download_file(url, save_path):
    print(f"Downloading {url}")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[✓] Saved → {save_path}")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

def download_esol():
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    raw_path = os.path.join(SAVE_DIR, "delaney_raw.csv")
    save_path = os.path.join(SAVE_DIR, "delaney-processed.csv")

    download_file(url, raw_path)

    import pandas as pd
    df = pd.read_csv(raw_path)

    # 只保留 smiles + solubility
    df = df[["smiles", "measured log solubility in mols per litre"]]

    # 重命名（建议统一）
    df.columns = ["smiles", "y"]

    df.to_csv(save_path, index=False)

    print(f"[✓] Cleaned ESOL → {save_path} | size = {len(df)}")

def download_tox21():
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    gz_path = os.path.join(SAVE_DIR, "tox21.csv.gz")
    save_path = os.path.join(SAVE_DIR, "tox21.csv")

    download_file(url, gz_path)

    import pandas as pd
    import gzip
    import shutil

    # 解压
    with gzip.open(gz_path, 'rb') as f_in:
        with open(save_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    df = pd.read_csv(save_path)

    # 删除 mol_id（如果存在）
    if "mol_id" in df.columns:
        df = df.drop(columns=["mol_id"])

    # 确保 smiles 在第一列（更规范）
    cols = df.columns.tolist()
    if "smiles" in cols:
        cols.remove("smiles")
        df = df[["smiles"] + cols]

    df.to_csv(save_path, index=False)

    print(f"[✓] Cleaned Tox21 → {save_path} | size = {len(df)}")


def save_dataset(name, smiles, y):
    """
    保存为标准 CSV：
    columns: smiles, y
    """
    df = pd.DataFrame({
        "smiles": smiles,
        "y": y.flatten()
    })

    save_path = os.path.join(SAVE_DIR, f"{name}.csv")
    df.to_csv(save_path, index=False)
    print(f"[✓] Saved {name} → {save_path} | size = {len(df)}")


def load_and_save_bbbp():
    tasks, datasets, _ = dc.molnet.load_bbbp(
        featurizer='Raw',
        splitter=None
    )
    dataset = datasets[0]
    save_dataset("bbbp", dataset.ids, dataset.y)


def load_and_save_freesolv():
    tasks, datasets, _ = dc.molnet.load_freesolv(
        featurizer='Raw',
        splitter=None
    )
    dataset = datasets[0]
    save_dataset("freesolv", dataset.ids, dataset.y)


def load_and_save_lipo():
    tasks, datasets, _ = dc.molnet.load_lipo(
        featurizer='Raw',
        splitter=None
    )
    dataset = datasets[0]
    save_dataset("lipophilicity", dataset.ids, dataset.y)


def load_and_save_hiv():
    tasks, datasets, _ = dc.molnet.load_hiv(
        featurizer='Raw',
        splitter=None
    )
    dataset = datasets[0]
    save_dataset("hiv", dataset.ids, dataset.y)

def download_qm9():
    import deepchem as dc
    import pandas as pd
    import os

    print("Downloading QM9...")

    tasks, datasets, _ = dc.molnet.load_qm9(
        featurizer='Raw',
        splitter=None
    )

    dataset = datasets[0]

    df = pd.DataFrame({
        "smiles": dataset.ids
    })

    save_path = os.path.join(SAVE_DIR, "qm9_smiles.csv")
    df.to_csv(save_path, index=False)

    print(f"[✓] Saved QM9 → {save_path} | size = {len(df)}")

# 在 data_download.py 里加

def main():
    print("Downloading datasets...\n")

    download_qm9()

    print("\n🎉 All datasets ready!")


if __name__ == "__main__":
    main()

#if __name__ == "__main__":
    #print("Downloading and saving MoleculeNet datasets...\n")

    #load_and_save_bbbp()
    #load_and_save_freesolv()
    #load_and_save_lipo()
    #load_and_save_hiv()

    #print("\n🎉 All datasets downloaded and saved!")
