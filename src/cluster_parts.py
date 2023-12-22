import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from pathlib import Path
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster import hierarchy
from sklearn.metrics.pairwise import cosine_similarity
import os

class BodyPart:
    def __init__(
            self,
            part_df: pd.DataFrame,
            db_path: Path = None,
            part_col='EPart',
    ):
        if db_path is not None:
            self.client = chromadb.PersistentClient(path=str(db_path))
        else:
            self.client = chromadb.Client()
        self.collection = self.client.create_collection("collection0")
        self.key_ele = part_df[part_col].dropna().unique().tolist()
        print('start creating vector database')
        self.collection.add(
            documents=self.key_ele,  # we embed for you, or bring your own
            metadatas=[{'id': i} for i, k in enumerate(self.key_ele)],  # filter on arbitrary metadata!
            ids=[f'i{i + 1}' for i in range(len(self.key_ele))]  # must be unique for each doc
        )

        print('vector database created')
    def cluster(self, distance_threshold: float = .3):
        self.clustering = AgglomerativeClustering(
            metric='cosine',
            distance_threshold=distance_threshold,
            n_clusters=None,
            linkage='average').fit(np.array(self.collection.peek(len(self.key_ele))['embeddings']))
        self.labels = self.clustering.labels_
        self.results = pd.DataFrame({'KeyElement': self.collection.peek(len(self.key_ele))['documents'], 'cluster': self.labels})
        rcount = self.results['cluster'].value_counts()
        self.results.index = self.results['cluster'].to_list()
        self.results['count'] = self.results['cluster'].apply(lambda e: rcount.loc[e])
        self.results.sort_values(['count', 'cluster'], ascending=False, inplace=True)
        return self.results

    def sort(self, k: int) -> pd.DataFrame:
        '''
        sort a list of texts based on hierarchical cluster
        :param texts:
        :param k: Number of Clusters
        :return:
        '''
        X = np.array(self.collection.peek(len(self.key_ele))['embeddings'])
        texts = pd.Series(self.key_ele)
        n_response = len(texts)

        # X = (X>0)*1
        Z = linkage(X, 'average', 'cosine')
        sort_order = [idx for idx in hierarchy.leaves_list(Z) if idx < n_response]
        cos_sim = cosine_similarity(X, X)
        maxp = np.argmax(cos_sim, axis=1)
        maxv = np.max(cos_sim, axis=1)
        fl = fcluster(Z, k, criterion='maxclust')
        summary_df = pd.DataFrame({
            "texts": texts.to_list(),
            "cluster_id": fl
        })
        summary_df = summary_df.loc[sort_order]
        return summary_df
