from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Tuple
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
          
            documents_clean = [str(doc) for doc in documents]
            
          
            empty_docs = [i for i, doc in enumerate(documents_clean) if not doc.strip()]
            if empty_docs:
                print(f"⚠️  Ditemukan {len(empty_docs)} dokumen kosong, akan diabaikan")
            
          
            self.vectorizer = CountVectorizer(
                lowercase=False 
            )
            
            print("📊 Membuat vocabulary dan matrix...")
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