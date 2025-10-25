# UTS_PI_IR_System.py
# Information Retrieval System dengan Whoosh & Cosine Similarity
# Untuk UTS Praktikum Penelusuran Informasi

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import whoosh.index as index
from whoosh import fields, index, qparser, scoring
from whoosh.analysis import StandardAnalyzer
import os
import ast
from typing import List, Dict, Tuple
import json
import sys

class DataLoader:
    """Class untuk memuat dan mempersiapkan data"""
    
    def __init__(self):
        self.df = None
        self.is_loaded = False
    
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
            self.df['judul_text'] = self.df['judul'].astype(str)
            self.df['konten_text'] = self.df['konten'].astype(str)
            self.df['full_text'] = self.df['judul_text'] + " " + self.df['konten_text']

            # Tambahkan doc_id jika belum ada
            if 'doc_id' not in self.df.columns:
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

class BowRepresentation:
    """Class untuk representasi Bag of Words"""
    
    def __init__(self):
        self.vectorizer = None
        self.bow_matrix = None
        self.feature_names = None
        self.is_created = False
    
    def create_bow(self, documents: List[str]):
        """Membuat representasi BoW dari dokumen"""
        print("🔄 Creating Bag of Words representation...")
        
        try:
            # Pastikan semua documents adalah string
            documents_clean = [str(doc) for doc in documents]
            
            # Cek jika ada dokumen yang kosong setelah cleaning
            empty_docs = [i for i, doc in enumerate(documents_clean) if not doc.strip()]
            if empty_docs:
                print(f"⚠️  Ditemukan {len(empty_docs)} dokumen kosong, akan diabaikan")
            
            # Gunakan tokenizer default karena data sudah berupa text yang dipisah spasi
            self.vectorizer = CountVectorizer(
                lowercase=False  # Karena sudah lowercase di preprocessing
            )
            
            self.bow_matrix = self.vectorizer.fit_transform(documents_clean)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            print(f"✅ BoW created: {self.bow_matrix.shape[0]} docs, {self.bow_matrix.shape[1]} terms")
            self.is_created = True
            return self.bow_matrix
        except Exception as e:
            print(f"❌ Error creating BoW: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_query_vector(self, query: str):
        """Transform query menjadi vector"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer belum dibuat. Panggil create_bow() terlebih dahulu.")
        
        return self.vectorizer.transform([query])

class WhooshIndexer:
    """Class untuk indexing dengan Whoosh - FIXED VERSION"""
    
    def __init__(self, index_dir: str = "whoosh_index"):
        self.index_dir = index_dir
        self.schema = None
        self.ix = None
        self.is_built = False
    
    def create_schema(self):
        """Membuat schema untuk index Whoosh"""
        # Gunakan StandardAnalyzer
        analyzer = StandardAnalyzer()
        
        self.schema = fields.Schema(
            doc_id=fields.ID(stored=True, unique=True),
            judul=fields.TEXT(stored=True, analyzer=analyzer),
            konten=fields.TEXT(stored=True, analyzer=analyzer),
            full_text=fields.TEXT(stored=True, analyzer=analyzer),
            dataset=fields.ID(stored=True)
        )
        return self.schema
    
    def build_index(self, df: pd.DataFrame):
        """Membangun index dari dataframe"""
        print("🔄 Building Whoosh index...")
        
        try:
            # Buat directory jika belum ada
            if not os.path.exists(self.index_dir):
                os.mkdir(self.index_dir)
            
            # Buat schema
            self.create_schema()
            
            # Buat index
            self.ix = index.create_in(self.index_dir, self.schema)
            
            # Index documents
            writer = self.ix.writer()
            
            for idx, row in df.iterrows():
                writer.add_document(
                    doc_id=str(row['doc_id']),
                    judul=row['judul_text'],
                    konten=row['konten_text'],
                    full_text=row['full_text'],
                    dataset=row['dataset']
                )
            
            writer.commit()
            print(f"✅ Whoosh index built: {self.ix.doc_count()} documents")
            self.is_built = True
            return self.ix
        except Exception as e:
            print(f"❌ Error building Whoosh index: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def search(self, query: str, limit: int = 10):
        """Search dengan Whoosh - FIXED VERSION"""
        if self.ix is None:
            raise ValueError("Index belum dibuat. Panggil build_index() terlebih dahulu.")
        
        try:
            with self.ix.searcher() as searcher:
                # Buat parser untuk query
                query_parser = qparser.MultifieldParser(["judul", "konten", "full_text"], self.ix.schema)
                parsed_query = query_parser.parse(query)
                
                # Lakukan pencarian
                results = searcher.search(parsed_query, limit=limit)
                
                # Convert results to list of dictionaries immediately while searcher is open
                results_list = []
                for result in results:
                    results_list.append({
                        'doc_id': result['doc_id'],
                        'judul': result['judul'],
                        'konten': result['konten'],
                        'full_text': result['full_text'],
                        'dataset': result['dataset'],
                        'score': result.score
                    })
                
                return results_list
        except Exception as e:
            print(f"❌ Error in Whoosh search: {e}")
            import traceback
            traceback.print_exc()
            return []

class CosineRanker:
    """Class untuk ranking dengan Cosine Similarity"""
    
    def __init__(self):
        self.bow_model = None
        self.df = None
        self.doc_vectors = None
        self.is_initialized = False
    
    def initialize(self, bow_model: BowRepresentation, df: pd.DataFrame):
        """Initialize cosine ranker dengan bow model dan data"""
        self.bow_model = bow_model
        self.df = df
        self.doc_vectors = bow_model.bow_matrix
        self.is_initialized = True
        print("✅ Cosine Ranker initialized")
    
    def rank_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Ranking dokumen berdasarkan cosine similarity dengan query"""
        if not self.is_initialized:
            raise ValueError("CosineRanker belum diinisialisasi. Panggil initialize() terlebih dahulu.")
        
        try:
            # Transform query menjadi vector
            query_vector = self.bow_model.get_query_vector(query)
            
            # Hitung cosine similarity
            similarities = cosine_similarity(query_vector, self.doc_vectors)
            
            # Dapatkan top-k documents
            similarity_scores = similarities[0]
            top_indices = np.argsort(similarity_scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarity_scores[idx] > 0:  # Hanya dokumen dengan similarity > 0
                    doc_id = self.df.iloc[idx]['doc_id']
                    judul = self.df.iloc[idx]['judul_text'][:100] + "..." if len(self.df.iloc[idx]['judul_text']) > 100 else self.df.iloc[idx]['judul_text']
                    dataset = self.df.iloc[idx]['dataset']
                    
                    results.append({
                        'rank': len(results) + 1,
                        'doc_id': doc_id,
                        'score': float(similarity_scores[idx]),
                        'judul': judul,
                        'dataset': dataset
                    })
            
            return results
        except Exception as e:
            print(f"❌ Error in cosine ranking: {e}")
            return []
    
    def hybrid_search(self, whoosh_results, query: str, top_k: int = 5):
        """Hybrid search: gabungkan Whoosh results dengan cosine similarity - FIXED VERSION"""
        if not self.is_initialized:
            raise ValueError("CosineRanker belum diinisialisasi. Panggil initialize() terlebih dahulu.")
        
        try:
            if not whoosh_results:
                print("⚠️ Whoosh tidak menemukan hasil")
                return []
            
            # Dapatkan doc_ids dari Whoosh results (sudah dalam bentuk dictionary)
            whoosh_doc_ids = [int(r['doc_id']) for r in whoosh_results]
            
            # Filter dokumen yang ada di whoosh results
            whoosh_indices = []
            for idx, doc_id in enumerate(self.df['doc_id']):
                if doc_id in whoosh_doc_ids:
                    whoosh_indices.append(idx)
            
            if not whoosh_indices:
                print("⚠️ Tidak ada dokumen yang cocok untuk hybrid search")
                return []
            
            # Hitung cosine similarity untuk dokumen yang difilter
            query_vector = self.bow_model.get_query_vector(query)
            filtered_vectors = self.doc_vectors[whoosh_indices]
            
            similarities = cosine_similarity(query_vector, filtered_vectors)
            similarity_scores = similarities[0]
            
            # Gabungkan dengan whoosh scores
            combined_results = []
            for i, idx in enumerate(whoosh_indices):
                doc_id = self.df.iloc[idx]['doc_id']
                # Cari whoosh score dari results
                whoosh_score = 0
                for result in whoosh_results:
                    if result['doc_id'] == str(doc_id):
                        whoosh_score = result['score']
                        break
                
                cosine_score = similarity_scores[i]
                
                # Combined score (bisa di-weight sesuai kebutuhan)
                combined_score = 0.3 * whoosh_score + 0.7 * cosine_score
                
                combined_results.append({
                    'doc_id': doc_id,
                    'whoosh_score': whoosh_score,
                    'cosine_score': cosine_score,
                    'combined_score': combined_score,
                    'judul': self.df.iloc[idx]['judul_text'],
                    'dataset': self.df.iloc[idx]['dataset']
                })
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return combined_results[:top_k]
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
            import traceback
            traceback.print_exc()
            return []

class IRSystemCLI:
    """Command Line Interface untuk Information Retrieval System"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.bow_model = BowRepresentation()
        self.indexer = WhooshIndexer()
        self.cosine_ranker = CosineRanker()
        self.df = None
        self.is_system_ready = False
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*50)
        print("    INFORMATION RETRIEVAL SYSTEM")
        print("="*50)
        print("[1] Load & Index Dataset")
        print("[2] Search Query")
        print("[3] Exit")
        print("="*50)
        
        if self.is_system_ready:
            print("✅ Status: Sistem siap untuk pencarian")
        else:
            print("❌ Status: Silakan load dataset terlebih dahulu (Menu 1)")
    
    def load_and_index_dataset(self):
        """Menu 1: Load & Index Dataset"""
        print("\n📂 LOAD & INDEX DATASET")
        print("-" * 30)
        
        file_path = input("Masukkan path file dataset (default: step_data\\step6_detokenized.csv): ").strip()
        if not file_path:
            file_path = r"step_data\step6_detokenized.csv"
        
        print(f"\n🔄 Loading data dari: {file_path}")
        self.df = self.data_loader.load_processed_data(file_path)
        
        if self.df is None:
            print("❌ Gagal memuat data!")
            return False
        
        print("\n🔄 Membuat Bag of Words representation...")
        documents_text = self.df['full_text'].tolist()
        bow_matrix = self.bow_model.create_bow(documents_text)
        
        if bow_matrix is None:
            print("❌ Gagal membuat BoW representation!")
            return False
        
        print("\n🔄 Membangun Whoosh index...")
        ix = self.indexer.build_index(self.df)
        
        if ix is None:
            print("❌ Gagal membangun Whoosh index!")
            return False
        
        print("\n🔄 Menginisialisasi Cosine Ranker...")
        self.cosine_ranker.initialize(self.bow_model, self.df)
        
        print("\n📈 SYSTEM PERFORMANCE SUMMARY")
        print("=" * 40)
        print(f"✅ Total Documents: {len(self.df):,}")
        if self.bow_model.feature_names is not None:
            print(f"✅ Vocabulary Size: {len(self.bow_model.feature_names):,}")
        print(f"✅ BoW Matrix: {bow_matrix.shape} (docs x features)")
        
        # Check memory usage
        if hasattr(bow_matrix, 'data'):
            bow_size = (bow_matrix.data.nbytes + bow_matrix.indptr.nbytes + bow_matrix.indices.nbytes) / (1024 * 1024)
            print(f"✅ BoW Memory: {bow_size:.2f} MB")
        
        self.is_system_ready = True
        print("\n🎉 SISTEM BERHASIL DILOAD DAN SIAP DIGUNAKAN!")
        
        # Tampilkan contoh query yang bisa dicoba
        print("\n💡 CONTOH QUERY YANG BISA DICOBA:")
        print("   • 'drone militer afghanistan'")
        print("   • 'apartemen jakarta harga'")
        print("   • 'teknologi artificial intelligence'")
        print("   • 'presiden obama'")
        print("   • 'pendidikan tinggi'")
        
        return True
    
    def search_query(self):
        """Menu 2: Search Query"""
        if not self.is_system_ready:
            print("❌ Sistem belum siap! Silakan load dataset terlebih dahulu (Menu 1)")
            return
        
        print("\n🔍 SEARCH QUERY")
        print("-" * 30)
        query = input("Masukkan query: ").strip()
        
        if not query:
            print("❌ Query tidak boleh kosong!")
            return
        
        print(f"\nMencari: '{query}'")
        print("-" * 50)
        
        print("Pilih metode search:")
        print("[1] Whoosh Search")
        print("[2] Cosine Similarity")
        print("[3] Hybrid Search")
        
        try:
            choice = int(input("Pilihan (1-3): "))
        except:
            print("❌ Pilihan tidak valid!")
            return
        
        if choice == 1:
            self._whoosh_search(query)
        elif choice == 2:
            self._cosine_search(query)
        elif choice == 3:
            self._hybrid_search(query)
        else:
            print("❌ Pilihan tidak valid!")
    
    def _whoosh_search(self, query: str):
        """Whoosh search only - FIXED VERSION"""
        try:
            results = self.indexer.search(query, limit=5)
            
            print(f"\n🔍 WHOOSH SEARCH RESULTS ({len(results)} documents):")
            if len(results) == 0:
                print("   Tidak ada hasil yang ditemukan")
                return
                
            for i, result in enumerate(results):
                print(f"\n{i+1}. DocID: {result['doc_id']}")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Dataset: {result['dataset']}")
                print(f"   Judul: {result['judul'][:80]}...")
        except Exception as e:
            print(f"❌ Error dalam Whoosh search: {e}")
            import traceback
            traceback.print_exc()
    
    def _cosine_search(self, query: str):
        """Cosine similarity search only"""
        try:
            results = self.cosine_ranker.rank_documents(query, top_k=5)
            
            print(f"\n📊 COSINE SIMILARITY RESULTS ({len(results)} documents):")
            if len(results) == 0:
                print("   Tidak ada hasil yang ditemukan")
                return
                
            for result in results:
                print(f"\n{result['rank']}. DocID: {result['doc_id']}")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Dataset: {result['dataset']}")
                print(f"   Judul: {result['judul']}")
        except Exception as e:
            print(f"❌ Error dalam Cosine search: {e}")
    
    def _hybrid_search(self, query: str):
        """Hybrid search - FIXED VERSION"""
        try:
            print("🔍 Mencari dengan Whoosh...")
            whoosh_results = self.indexer.search(query, limit=10)
            
            if not whoosh_results:
                print("❌ Whoosh tidak menemukan hasil, hybrid search dibatalkan")
                return
            
            print("📊 Menghitung cosine similarity...")
            hybrid_results = self.cosine_ranker.hybrid_search(whoosh_results, query, top_k=5)
            
            print(f"\n🎯 HYBRID SEARCH RESULTS ({len(hybrid_results)} documents):")
            if len(hybrid_results) == 0:
                print("   Tidak ada hasil yang ditemukan")
                return
                
            for i, result in enumerate(hybrid_results):
                print(f"\n{i+1}. DocID: {result['doc_id']}")
                print(f"   Combined Score: {result['combined_score']:.4f}")
                print(f"   (Whoosh: {result['whoosh_score']:.4f}, Cosine: {result['cosine_score']:.4f})")
                print(f"   Dataset: {result['dataset']}")
                print(f"   Judul: {result['judul'][:80]}...")
        except Exception as e:
            print(f"❌ Error dalam Hybrid search: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Run CLI application"""
        print("🚀 INFORMATION RETRIEVAL SYSTEM")
        print("   UTS Praktikum Penelusuran Informasi")
        print("   Sistem CLI dengan Whoosh & Cosine Similarity")
        
        while True:
            self.display_menu()
            
            try:
                choice = int(input("\nPilih menu (1-3): "))
            except ValueError:
                print("❌ Input harus angka!")
                continue
            
            if choice == 1:
                self.load_and_index_dataset()
            elif choice == 2:
                self.search_query()
            elif choice == 3:
                print("\n👋 Terima kasih telah menggunakan Information Retrieval System!")
                break
            else:
                print("❌ Pilihan tidak valid! Silakan pilih 1-3.")

def main():
    """Main function"""
    system = IRSystemCLI()
    system.run()

if __name__ == "__main__":
    main()