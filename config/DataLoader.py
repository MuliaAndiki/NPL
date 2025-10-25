import os
import pandas as pd
class DataLoader:
    """Class untuk memuat dan mempersiapkan data"""
    
    def __init__(self):
        self.df = None
        self.is_loaded = False
    
    def show_progress(self, current, total, prefix="", suffix="", length=50):
        """Menampilkan progress bar"""
        percent = current / total
        filled_length = int(length * percent)
        bar = "█" * filled_length + "─" * (length - filled_length)
        percent_display = percent * 100
        print(f"\r{prefix} |{bar}| {percent_display:.1f}% {suffix}", end="", flush=True)
        if current == total:
            print()
    
    def load_processed_data(self, file_path: str = r"step_data\step6_detokenized.csv"):
        """Muat data hasil stemming yang sudah di-detokenized"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ File tidak ditemukan di path: {file_path}")
                return None

            print(f"📂 Membaca file: {file_path}")
            self.df = pd.read_csv(file_path)
            print(f"✅ Data loaded: {len(self.df):,} baris")
            
            print("\n🔍 Preview 2 baris teratas:")
            print(self.df.head(2))
            print("\n📑 Kolom yang terbaca:", list(self.df.columns))

            # Bersihkan data dari NaN values
            print("🧹 Membersihkan data dari NaN values...")
            initial_count = len(self.df)
            self.df = self.df.dropna(subset=['judul', 'konten'])
            cleaned_count = len(self.df)
            
            if initial_count > cleaned_count:
                print(f"   - Dihapus {initial_count - cleaned_count} baris dengan nilai NaN")
            
            # Untuk data yang sudah di-detokenized, langsung gunakan sebagai text
            print("🔄 Memproses teks...")
            total_rows = len(self.df)
            
            # Process dengan progress bar
            judul_texts = []
            konten_texts = []
            full_texts = []
            
            for i, (idx, row) in enumerate(self.df.iterrows()):
                self.show_progress(i + 1, total_rows, "🔄 Memproses dokumen", f"{i+1}/{total_rows}")
                
                judul_text = str(row['judul'])
                konten_text = str(row['konten'])
                full_text = judul_text + " " + konten_text
                
                judul_texts.append(judul_text)
                konten_texts.append(konten_text)
                full_texts.append(full_text)
            
            self.df['judul_text'] = judul_texts
            self.df['konten_text'] = konten_texts
            self.df['full_text'] = full_texts

            # Tambahkan doc_id jika belum ada
            if 'doc_id' not in self.df.columns:
                print("🔢 Menambahkan doc_id...")
                self.df['doc_id'] = range(1, len(self.df) + 1)

            # Summary dataset
            if 'dataset' in self.df.columns:
                print("\n📊 Dataset Summary:")
                for dataset in self.df['dataset'].unique():
                    count = len(self.df[self.df['dataset'] == dataset])
                    print(f"   - {dataset}: {count} dokumen")
            else:
                # Jika tidak ada kolom dataset, tambahkan default
                self.df['dataset'] = 'merged_data'
                print(f"   - Dataset: {len(self.df)} dokumen (merged)")

            print(f"\n✅ Data siap digunakan:")
            print(f"   - Total dokumen: {len(self.df):,}")
            print(f"   - Sample judul: {self.df.iloc[0]['judul_text'][:100]}...")
            print(f"   - Sample konten: {self.df.iloc[0]['konten_text'][:100]}...")
            
            self.is_loaded = True
            return self.df

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None