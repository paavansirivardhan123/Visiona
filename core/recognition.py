import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
import time

class FeatureDB:
    """
    Lightweight, real-time custom object recognizer (Vector Memory).
    Uses MobileNetV3 to extract feature embeddings from object crops.
    """
    def __init__(self, db_path="database"):
        self.db_path = db_path
        self._db = {}  # Format: { alias: tensor(N, embedding_dim) }
        
        print("  [Memory] Initializing Vector Embeddings Module (MobileNetV3)...")
        # Use weights explicitly to suppress warnings on new PyTorch versions
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights).eval()
        
        # Remove the classification head to just get the feature vector
        # MobileNet features usually output [B, 576, 1, 1]
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Preload all existing databases on disk
        self._load_all()

    def _load_all(self):
        """Loads embeddings for all objects currently saved in the database directory."""
        if not os.path.exists(self.db_path):
            return
            
        aliases = [d for d in os.listdir(self.db_path) if os.path.isdir(os.path.join(self.db_path, d))]
        for alias in aliases:
            self.load_alias(alias)

    def load_alias(self, alias: str):
        """Processes a folder of images for a specific alias and caches their embeddings."""
        dir_path = os.path.join(self.db_path, alias)
        if not os.path.exists(dir_path):
            return
            
        print(f"  [Memory] Indexing features for custom object '{alias}'...")
        embeddings = []
        for file in os.listdir(dir_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(dir_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            emb = self._get_embedding(img_rgb)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            # Shape: N x Embedding_Dim
            self._db[alias] = torch.cat(embeddings, dim=0)

    def _get_embedding(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Returns a normalized 1D embedding tensor for a single RGB image crop."""
        try:
            tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(tensor)
                # Output is [1, C, 1, 1], flatten to [1, C]
                features = torch.flatten(features, 1)
                # L2 Normalize
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features
        except Exception as e:
            print(f"  [Memory] Error extracting embedding: {e}")
            return None

    def match(self, img_bgr: np.ndarray, threshold: float = 0.82) -> str:
        """
        Attempts to match an unknown cropped image against the database.
        Returns the alias name if similarity > threshold, otherwise None.
        """
        if not self._db or img_bgr is None or img_bgr.size == 0:
            return None
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        emb = self._get_embedding(img_rgb)
        
        if emb is None:
            return None
            
        best_match = None
        best_score = -1.0
        
        # Compare embedding to all cached alias embeddings
        for alias, db_embs in self._db.items():
            # db_embs shape is [N, C], emb is [1, C]
            # Cosine similarity is the dot product because they are L2-normalized
            similarities = torch.mm(emb, db_embs.transpose(0, 1))
            
            max_sim = similarities.max().item()
            if max_sim > best_score:
                best_score = max_sim
                best_match = alias
                
        if best_match and best_score >= threshold:
            return best_match
            
        return None

# Global Singleton
feature_db = FeatureDB()
