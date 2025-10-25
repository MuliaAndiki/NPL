import os
import pandas as pd
from multiprocessing import Pool, cpu_count, freeze_support
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tqdm import tqdm

_stemmer = None


def init_stemmer():
    global _stemmer
    _stemmer = StemmerFactory().create_stemmer()


def stemming_worker(text: str):
    global _stemmer
    if not isinstance(text, str):
        return ""

    text = text.replace("[", "").replace("]", "").replace("'", "")
    text = text.replace(",", " ")

    try:
        return _stemmer.stem(text)
    except Exception as e:
        return f"ERROR: {e}"


def process_batch(df_batch: pd.DataFrame, num_cores: int):
    """Memproses satu batch data secara paralel."""
    with Pool(num_cores, initializer=init_stemmer) as pool:
        judul_iter = pool.imap_unordered(stemming_worker, df_batch["judul"])
        konten_iter = pool.imap_unordered(stemming_worker, df_batch["konten"])

        judul_stem = list(
            tqdm(
                judul_iter,
                total=len(df_batch),
                desc="🔤 Stemming judul",
                ncols=100,
                leave=False,
            )
        )
        konten_stem = list(
            tqdm(
                konten_iter,
                total=len(df_batch),
                desc="📰 Stemming konten",
                ncols=100,
                leave=False,
            )
        )

    df_batch["judul"] = judul_stem
    df_batch["konten"] = konten_stem
    return df_batch


def step5_stemming_parallel_batch(
    input_file="step_data/step4_stopword.csv",
    output_file="step_data/step5_stemming_token_parallel_batch.csv",
    batch_size=None,
):
    """Melakukan stemming paralel per batch secara efisien."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"❌ File {input_file} tidak ditemukan.")
        return

    total_cores = cpu_count()
    num_cores = max(1, total_cores - 1)

    if batch_size is None:
        if total_cores <= 4:
            batch_size = 1000
        elif total_cores <= 8:
            batch_size = 3000
        else:
            batch_size = 5000

    print(f"🧠 Menggunakan {num_cores} dari {total_cores} core CPU...")
    print(f"⚙️ Ukuran batch adaptif: {batch_size} baris per iterasi\n")

    with open(input_file, encoding="utf-8", errors="ignore") as f:
        total_rows = sum(1 for _ in f) - 1
    print(f"📊 Total baris dalam dataset: {total_rows:,}")

    batch_iter = pd.read_csv(
        input_file,
        chunksize=batch_size,
        encoding="utf-8",
        on_bad_lines="skip",
    )

    is_first = True
    for i, df_chunk in enumerate(batch_iter, start=1):
        start_idx = (i - 1) * batch_size + 1
        end_idx = start_idx + len(df_chunk) - 1
        print(f"\n🔹 Memproses batch {i}: baris {start_idx:,}–{end_idx:,} ...")

        if not {"judul", "konten"}.issubset(df_chunk.columns):
            print("⚠️ Batch dilewati: kolom 'judul' atau 'konten' tidak ditemukan.")
            continue

        df_processed = process_batch(df_chunk, num_cores)

        df_processed.to_csv(
            output_file,
            mode="a",
            index=False,
            header=is_first,
            encoding="utf-8",
        )
        is_first = False

        print(f"✅ Batch {i} selesai → disimpan ke: {output_file}")

    print("\n🎉 Semua batch selesai diproses!")
    print(f"📁 File hasil akhir: {output_file}")


if __name__ == "__main__":
    freeze_support() 
    step5_stemming_parallel_batch()
