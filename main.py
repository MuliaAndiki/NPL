import time
from config.DataLoader import DataLoader
from config.BowRepresentation import BowRepresentation
from config.WhoosheIndexer import WhooshIndexer
from config.Cosine import CosineRanker

class IRSystemCLI:
    
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
        start_time = time.time()
        
        self.df = self.data_loader.load_processed_data(file_path)
        
        if self.df is None:
            print("❌ Gagal memuat data!")
            return False
        
        load_time = time.time() - start_time
        print(f"⏱️  Waktu loading data: {load_time:.2f} detik")
        
        print("\n🔄 Membuat Bag of Words representation...")
        bow_start = time.time()
        documents_text = self.df['full_text'].tolist()
        bow_matrix = self.bow_model.create_bow(documents_text)
        bow_time = time.time() - bow_start
        
        if bow_matrix is None:
            print("❌ Gagal membuat BoW representation!")
            return False
        
        print(f"⏱️  Waktu membuat BoW: {bow_time:.2f} detik")
        
        print("\n🔄 Membangun Whoosh index...")
        index_start = time.time()
        ix = self.indexer.build_index(self.df)
        index_time = time.time() - index_start
        
        if ix is None:
            print("❌ Gagal membangun Whoosh index!")
            return False
        
        print(f"⏱️  Waktu membangun index: {index_time:.2f} detik")
        
        print("\n🔄 Menginisialisasi Cosine Ranker...")
        self.cosine_ranker.initialize(self.bow_model, self.df)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  TOTAL WAKTU: {total_time:.2f} detik")
        
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
    
    def ask_show_content(self):
        """Tanya user apakah ingin menampilkan konten"""
        print("\n📄 Tampilkan konten dokumen?")
        print("[1] Ya, tampilkan konten")
        print("[2] Tidak, hanya judul saja")
        
        try:
            choice = int(input("Pilihan (1-2): "))
            return choice == 1
        except:
            print("❌ Pilihan tidak valid! Hanya menampilkan judul.")
            return False
    
    def display_search_results(self, results, show_content=False):
        """Menampilkan hasil pencarian dengan opsi konten"""
        for i, result in enumerate(results):
            print(f"\n{'='*60}")
            print(f"📄 HASIL {i+1}")
            print(f"{'='*60}")
            print(f"📋 DocID: {result['doc_id']}")
            print(f"⭐ Score: {result['score']:.4f}")
            print(f"📂 Dataset: {result['dataset']}")
            print(f"📝 Judul: {result['judul']}")
            
            if show_content and 'konten' in result:
                print(f"\n📖 Konten:")
                print(f"   {result['konten']}")
            
            print(f"{'='*60}")
    
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
        
        # Tanya apakah ingin menampilkan konten
        show_content = self.ask_show_content()
        
        print("\nPilih metode search:")
        print("[1] Whoosh Search")
        print("[2] Cosine Similarity")
        print("[3] Hybrid Search")
        
        try:
            choice = int(input("Pilihan (1-3): "))
        except:
            print("❌ Pilihan tidak valid!")
            return
        
        search_start = time.time()
        
        if choice == 1:
            self._whoosh_search(query, show_content)
        elif choice == 2:
            self._cosine_search(query, show_content)
        elif choice == 3:
            self._hybrid_search(query, show_content)
        else:
            print("❌ Pilihan tidak valid!")
            return
        
        search_time = time.time() - search_start
        print(f"\n⏱️  Waktu pencarian: {search_time:.2f} detik")
    
    def _whoosh_search(self, query: str, show_content: bool = False):
        """Whoosh search only - FIXED VERSION"""
        try:
            results = self.indexer.search(query, limit=5)
            
            print(f"\n🔍 WHOOSH SEARCH RESULTS ({len(results)} documents):")
            if len(results) == 0:
                print("   Tidak ada hasil yang ditemukan")
                return
            
            self.display_search_results(results, show_content)
                
        except Exception as e:
            print(f"❌ Error dalam Whoosh search: {e}")
            import traceback
            traceback.print_exc()
    
    def _cosine_search(self, query: str, show_content: bool = False):
        """Cosine similarity search only"""
        try:
            results = self.cosine_ranker.rank_documents(query, top_k=5)
            
            print(f"\n📊 COSINE SIMILARITY RESULTS ({len(results)} documents):")
            if len(results) == 0:
                print("   Tidak ada hasil yang ditemukan")
                return
            
            self.display_search_results(results, show_content)
                
        except Exception as e:
            print(f"❌ Error dalam Cosine search: {e}")
    
    def _hybrid_search(self, query: str, show_content: bool = False):
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
            
            # Format results untuk display
            formatted_results = []
            for i, result in enumerate(hybrid_results):
                formatted_results.append({
                    'rank': i + 1,
                    'doc_id': result['doc_id'],
                    'score': result['combined_score'],
                    'judul': result['judul'],
                    'konten': result['konten'],
                    'dataset': result['dataset'],
                    'details': f"(Whoosh: {result['whoosh_score']:.4f}, Cosine: {result['cosine_score']:.4f})"
                })
            
            self.display_search_results(formatted_results, show_content)
                
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