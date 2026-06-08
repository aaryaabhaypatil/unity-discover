from PIL import Image
import torch
from pathlib import Path
import json
import numpy as np
import open_clip
import torch
from pathlib import Path
import re

def to_text(path):
    s = Path(path).stem
    m = re.match(r'[a-zA-Z]+', s)
    if m:
        val = m.group(0)
        if len(val) > 1:
            return val.lower()

    return None


def match_rank(p, q):
    if not p or not q:
        return (-1, -1)
    
    if p == q:
        return (1, len(p))

    total = min(len(p), len(q))
    for i in range(min(len(p), len(q))):
        if p[i] != q[i]:
            return (0.8 * i/total, i)
        
    return (0.8, total)


class EmbeddingSet: 
    def __init__(self, model, paths, embeddings):
        self.model = model
        self.paths = paths

        embeddings = np.array(embeddings, dtype=np.float32)

        # normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms

        self.embeddings = embeddings

    def cosine_similarity(self, query_embedding):
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        sims = self.embeddings @ query_embedding
        return sims
    
    def top_k(self, query_embedding, k=5):
        sims = self.cosine_similarity(query_embedding)
        k = min(k, sims.shape[0])
        best_k = np.argsort(sims)[::-1][:k]      # highest first
        scores = sims[best_k]
        paths = [self.paths[int(i)] for i in best_k]
        return paths, scores.tolist()
    
    def label_search(self, q):
        if not q:
            return [], []
        q = q.lower()
        matches = [(match_rank(to_text(p), q), i) for i, p in enumerate(self.paths)]
        matches = [r for r in matches if r[0][1] >= 2]
        matches.sort(key=lambda x: x[0][0], reverse=True)
        paths = [self.paths[i] for _, i in matches]
        scores = [r[0][0] for r in matches]
        return paths, scores
    
    def remove(self, idx_2_remove: list[bool]):
        self.paths = [p for p, remove in zip(self.paths, idx_2_remove) if not remove]
        self.embeddings = self.embeddings[~np.array(idx_2_remove)]  

    def top_k_combined(self, query_embedding, query_label, alpha=0.1, k=5):
        # emb_paths, emb_scores = self.top_k(query_embedding, n_emb)
        n_lab = round(k * alpha)
        # print(f"Performing combined search with alpha={alpha}, giving {n_lab} label-based results and {k - n_lab} embedding-based results.")
        lab_paths, lab_scores = self.label_search(query_label)
        lab_paths = lab_paths[:n_lab]
        lab_scores = lab_scores[:n_lab]

        n_emb = k - len(lab_paths)
        emb_paths, emb_scores = self.top_k(query_embedding, n_emb)

        # print(f"Label search found: {len(lab_paths)} ")
        all_ps = zip(lab_paths + emb_paths, lab_scores + emb_scores)
        all_ps = sorted(all_ps, key=lambda x: x[1], reverse=True)
        paths, scores = zip(*all_ps)
        return paths[:k], scores[:k]


class EmbeddingDatabase:
    def __init__(self, db_path):
        self._models = {}
        self.embeddings = {}
        self.db_path = str(db_path)
    
    def load_json(self):
        path = Path(self.db_path + ".json") 
        if not path.exists():
            return
        
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                self._models = data.get('models', {})
                self.embeddings = data.get('embeddings', {})
    
    
    def save_json(self):
        with open(self.db_path + ".json", 'w') as f:
            database = {'models': self.models, 'embeddings': self.embeddings}
            json.dump(database, f)

    @property
    def models(self):
        # Gather all unique models used in the embeddings
        models = {}
        for path in self.embeddings.keys():
            for model in self.embeddings[path].keys():
                if model not in models:
                    models[model] = self._models.get(model, {})
        return models

    @staticmethod
    def embedding_datatypes():
        return {
            1: np.float32,
            2: np.float16,
            3: np.float64
        }, {
            np.float32: 1,
            np.float16: 2,
            np.float64: 3
        }

    def save_binary(self, vector_datatype=np.float32):
        # File structure:
        # {vector_datatype_id}[1 bytes]
        # {models_count}[4 bytes]
        # {model_info_length}[4 bytes] {model_info_json}[model_info_length bytes] x models_count
        #
        # {num_image_paths}[4 bytes] 
        # {num_bytes_path}[2 bytes] {path}[num_bytes_path bytes] x num_image_paths
        #
        # for each model:
        #   {num_embeddings}[4 bytes] 
        #   {size_of_embedding}[4 bytes] 
        #   for each embedding:
        #       {image_idx} [4 bytes]
        #   for num_embeddings: 
        #       {float}[4 bytes] x size_of_embedding

        _, dtype_to_id = self.embedding_datatypes()
        if vector_datatype not in dtype_to_id:
            raise ValueError(f"Unsupported vector datatype: {vector_datatype}")
        vector_datatype_id = dtype_to_id[vector_datatype]

        models = self.models.items()
        model_count = len(self.models)
        model_data = [json.dumps([id, info]).encode('utf-8') for id, info in models]
        model_data = [len(md).to_bytes(4, 'little') + md for md in model_data]
        model_section_bytes = model_count.to_bytes(4, 'little') + b''.join(model_data)


        image_path_to_idx = {path: idx for idx, path in enumerate(self.embeddings.keys())}
        images_paths = list(self.embeddings.keys())
        path_data = [path.encode('utf-8') for path in images_paths]
        path_data = [len(pd).to_bytes(2, 'little') + pd for pd in path_data]
        path_section_bytes = len(images_paths).to_bytes(4, 'little') + b''.join(path_data)


        embeddings_data = []
        for model, info in self.models.items():
            embeddings_for_model = [self.embeddings[path][model] for path in images_paths if model in self.embeddings[path]]
            embeddings_for_model = np.array(embeddings_for_model, dtype=vector_datatype)
            image_idxs = [image_path_to_idx[path] for path in images_paths if model in self.embeddings[path]]
            image_idxs = np.array(image_idxs, dtype=np.int32)
            num_embeddings = len(embeddings_for_model)
            if num_embeddings == 0:
                embeddings_data.append((0).to_bytes(4, 'little') + (0).to_bytes(4, 'little'))
            else:
                head = num_embeddings.to_bytes(4, 'little') + len(embeddings_for_model[0]).to_bytes(4, 'little')
                body = head + image_idxs.tobytes() + embeddings_for_model.tobytes()
                embeddings_data.append(body)
                
        with open(self.db_path + ".bin", 'wb') as f:
            f.write(vector_datatype_id.to_bytes(1, 'little'))
            f.write(model_section_bytes)
            f.write(path_section_bytes)
            f.write(b''.join(embeddings_data))


    def load_binary(self):
        # Read the binary format as defined in save_binary
        path = Path(self.db_path + ".bin")
        if not path.exists():
            print(f"No binary database found at {path}. Skipping load.")
            return
        
        with open(path, 'rb') as f:
            vector_datatype_id = int.from_bytes(f.read(1), 'little')
            id_to_dtype, _ = self.embedding_datatypes()
            if vector_datatype_id not in id_to_dtype:
                raise ValueError(f"Unsupported vector datatype ID: {vector_datatype_id}")
            vector_datatype = id_to_dtype[vector_datatype_id]

            model_count = int.from_bytes(f.read(4), 'little')

            model_ids = []
            for _ in range(model_count):
                model_info_length = int.from_bytes(f.read(4), 'little')
                model_info_json = f.read(model_info_length)
                model_id, model_info = json.loads(model_info_json)
                self._models[model_id] = model_info
                model_ids.append(model_id)

            num_image_paths = int.from_bytes(f.read(4), 'little')
            image_paths = []
            for _ in range(num_image_paths):
                path_length = int.from_bytes(f.read(2), 'little')
                path = f.read(path_length).decode('utf-8')
                image_paths.append(path)

            for i in range(model_count):
                model_id = model_ids[i]
                num_embeddings = int.from_bytes(f.read(4), 'little')
                embedding_size = int.from_bytes(f.read(4), 'little')
                if not num_embeddings:
                    break
                image_idxs = np.frombuffer(f.read(num_embeddings * 4), dtype=np.int32)
                embeddings = np.frombuffer(f.read(num_embeddings * embedding_size * np.dtype(vector_datatype).itemsize), dtype=vector_datatype)
                embeddings = embeddings.reshape((num_embeddings, embedding_size))
                
                for idx, image_idx in enumerate(image_idxs):
                    path = image_paths[image_idx]
                    if path not in self.embeddings:
                        self.embeddings[path] = {}
                    self.embeddings[path][model_id] = embeddings[idx].tolist()

        
    def add_model(self, model, model_info):
        self._models[str(model)] = model_info


    def add_embedding(self, path, model, embedding):
        if path not in self.embeddings:
            self.embeddings[path] = {}
        self.embeddings[path][str(model)] = embedding.tolist()

    def get_embeddings(self, model):
        model_id = str(model)
        if model_id not in self.models:
            raise ValueError(f"Model {model} not found in database.")
        
        paths = []
        embeddings = []
        for path, models in self.embeddings.items():
            if model_id in models:
                paths.append(path)
                embeddings.append(models[model_id])
        
        return EmbeddingSet(model, paths, embeddings)

class VectorEmbedder:
    def _embed_image(self, image):
        return []
    
    def _embed_text(self, text):
        return []
    

    @property
    def vectorSize(self):
        return 512
    

    def embed_image_batch(self, images):
        return self._embed_image_batch(images)
    

    def embed_image(self, image):
        img = Image.open(image).convert('RGB')
        return self._embed_image(img)
    

    def embed_text(self, text):
        return self._embed_text(text)
    
           
    def find_best_match(self, query_text, embeddings, k=5):
        query_embedding = np.asarray(self.embed_text(query_text), dtype=np.float32)
        doc_embeddings = np.asarray(embeddings["embeddings"], dtype=np.float32)

        # Safety normalization (in case some vectors aren't normalized)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
        doc_embeddings = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12)

        sims = doc_embeddings @ query_embedding # cosine similarity
        k = min(k, sims.shape[0])
        best_k = np.argsort(sims)[::-1][:k]      # highest first

        names = [embeddings["names"][int(i)] for i in best_k]
        scores = sims[best_k]
        return list(zip(scores.tolist(), names))
    
    def k_best(self, qv, allv, k=5):
        sims = allv @ qv
        k = min(k, sims.shape[0])
        best_k = np.argsort(sims)[::-1][:k]      # highest first
        scores = sims[best_k]
        return best_k, scores

    
    def __str__(self):
        return "Base Vector Embedder (no model)"
    
    @staticmethod
    def from_string(s):
        return VectorEmbedder()      

class Vit(VectorEmbedder):
    def __init__(self, model_name='ViT-B-32', pretrained='openai'):
        self.model_name = model_name
        self.pretrained = pretrained
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
    def load(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
        self.model.eval()
        self.model = self.model.to(self.device)
        self._size = self.embed_text("test").shape[0]
    
    def _embed_image(self, image):
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
        return embedding.squeeze().cpu().numpy()

    def _embed_image_batch(self, images):
        """Embed a batch of PIL images in a single forward pass."""
        img_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(img_tensors)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy()
    
    def _embed_text(self, text):
        text_tokens = open_clip.tokenize([text]).to(self.device)  # Tokenize the text
        with torch.no_grad():
            embedding = self.model.encode_text(text_tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize the embedding
        return embedding.squeeze().cpu().numpy()
    
    @property
    def vectorSize(self):
        return self._size
    
    def __str__(self):
        return f"Vit|{self.model_name}|{self.pretrained}"
    
    @staticmethod
    def from_string(s):
        parts = s.split('|')
        return Vit(model_name=parts[1], pretrained=parts[2])
    
