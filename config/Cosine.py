import pandas as pd
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from config.BowRepresentation import BowRepresentation
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CosineRanker:
    
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
            query_vector = self.bow_model.get_query_vector(query)
            
            similarities = cosine_similarity(query_vector, self.doc_vectors)
            
            similarity_scores = similarities[0]
            top_indices = np.argsort(similarity_scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarity_scores[idx] > 0:  
                    doc_id = self.df.iloc[idx]['doc_id']
                    judul = self.df.iloc[idx]['judul_text'][:100] + "..." if len(self.df.iloc[idx]['judul_text']) > 100 else self.df.iloc[idx]['judul_text']
                    konten = self.df.iloc[idx]['konten_text'][:200] + "..." if len(self.df.iloc[idx]['konten_text']) > 200 else self.df.iloc[idx]['konten_text']
                    dataset = self.df.iloc[idx]['dataset']
                    
                    results.append({
                        'rank': len(results) + 1,
                        'doc_id': doc_id,
                        'score': float(similarity_scores[idx]),
                        'judul': judul,
                        'konten': konten,
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
            
          
            whoosh_doc_ids = [int(r['doc_id']) for r in whoosh_results]
            
           
            whoosh_indices = []
            for idx, doc_id in enumerate(self.df['doc_id']):
                if doc_id in whoosh_doc_ids:
                    whoosh_indices.append(idx)
            
            if not whoosh_indices:
                print("⚠️ Tidak ada dokumen yang cocok untuk hybrid search")
                return []
            
          
            query_vector = self.bow_model.get_query_vector(query)
            filtered_vectors = self.doc_vectors[whoosh_indices]
            
            similarities = cosine_similarity(query_vector, filtered_vectors)
            similarity_scores = similarities[0]
            
           
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
                
               
                combined_score = 0.3 * whoosh_score + 0.7 * cosine_score
                
                combined_results.append({
                    'doc_id': doc_id,
                    'whoosh_score': whoosh_score,
                    'cosine_score': cosine_score,
                    'combined_score': combined_score,
                    'judul': self.df.iloc[idx]['judul_text'],
                    'konten': self.df.iloc[idx]['konten_text'][:200] + "..." if len(self.df.iloc[idx]['konten_text']) > 200 else self.df.iloc[idx]['konten_text'],
                    'dataset': self.df.iloc[idx]['dataset']
                })
            
          
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return combined_results[:top_k]
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
            import traceback
            traceback.print_exc()
            return []