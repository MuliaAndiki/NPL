import whoosh.index as index
from whoosh import fields, index, qparser, scoring
from whoosh.analysis import StandardAnalyzer
import os
import pandas as pd
class WhooshIndexer:
    """Class untuk indexing dengan Whoosh - FIXED VERSION"""
    
    def __init__(self, index_dir: str = "whoosh_index"):
        self.index_dir = index_dir
        self.schema = None
        self.ix = None
        self.is_built = False
    
    def show_progress(self, current, total, prefix="", suffix="", length=50):
        """Menampilkan progress bar"""
        percent = current / total
        filled_length = int(length * percent)
        bar = "█" * filled_length + "─" * (length - filled_length)
        percent_display = percent * 100
        print(f"\r{prefix} |{bar}| {percent_display:.1f}% {suffix}", end="", flush=True)
        if current == total:
            print()
    
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
            if not os.path.exists(self.index_dir):
                os.mkdir(self.index_dir)
            

            self.create_schema()
            
            self.ix = index.create_in(self.index_dir, self.schema)
            
            writer = self.ix.writer()
            total_docs = len(df)
            
            print("📝 Mengindex dokumen...")
            for idx, row in df.iterrows():
                if idx % 1000 == 0 or idx == total_docs - 1:
                    self.show_progress(idx + 1, total_docs, "📝 Mengindex", f"{idx+1}/{total_docs}")
                
                writer.add_document(
                    doc_id=str(row['doc_id']),
                    judul=row['judul_text'],
                    konten=row['konten_text'],
                    full_text=row['full_text'],
                    dataset=row['dataset']
                )
            
            print("\n💾 Menyimpan index...")
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
               
                query_parser = qparser.MultifieldParser(["judul", "konten", "full_text"], self.ix.schema)
                parsed_query = query_parser.parse(query)
                
               
                results = searcher.search(parsed_query, limit=limit)
                
               
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