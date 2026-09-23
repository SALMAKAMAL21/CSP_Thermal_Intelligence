"""
Modèle de fusion pour la classification niveau 2 (5 niveaux ordinaux 0-4) :
  - branche image siamoise (poids partagés) sur roi_test et roi_ref
  - branche tabulaire (MLP) sur delta_T normalisé
  - fusion par concaténation + tête de classification ordinale (CORAL)
"""

import torch
import torch.nn as nn
import torchvision.models as models

N_LEVELS = 5


class SharedCNNEncoder(nn.Module):
    """Backbone CNN partagé (poids identiques) pour roi_test et roi_ref -> réseau siamois."""

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # retire la dernière FC
        self.project = nn.Linear(backbone.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.project(x)


class TabularEncoder(nn.Module):
    """MLP pour la feature delta_T (et d'autres features tabulaires si ajoutées plus tard)."""

    def __init__(self, in_dim: int = 1, hidden_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class OrdinalHead(nn.Module):
    """Tête de classification ordinale façon CORAL : K-1 sorties binaires cumulatives
    au lieu d'un softmax K-classes classique. Respecte l'ordre naturel des niveaux
    de sévérité (0 < 1 < 2 < 3 < 4) au lieu de les traiter comme des catégories
    indépendantes.
    """

    def __init__(self, in_dim: int, n_levels: int = N_LEVELS):
        super().__init__()
        self.n_levels = n_levels
        # Une seule direction partagée (CORAL)
        self.shared = nn.Linear(in_dim, 1, bias=False)
        # Premier seuil libre + (n_levels-2) offsets forcés positifs (softplus) :
        # garantit b_0 > b_1 > ... > b_{K-2}, donc des seuils strictement
        # monotones. Sans ça, des biais libres peuvent finir désordonnés et
        # rendre certaines classes intermédiaires impossibles à prédire.
        self.first_bias = nn.Parameter(torch.tensor(0.0))
        self.offsets = nn.Parameter(torch.ones(n_levels - 2) * 0.5) if n_levels > 2 else None

    def forward(self, x):
        shared_score = self.shared(x)  # (batch, 1)
        if self.offsets is not None:
            deltas = torch.nn.functional.softplus(self.offsets)  # toujours > 0
            biases = self.first_bias - torch.cat([torch.zeros(1, device=x.device), torch.cumsum(deltas, dim=0)])
        else:
            biases = self.first_bias.unsqueeze(0)
        logits = shared_score + biases  # (batch, n_levels - 1)
        return logits

    @staticmethod
    def logits_to_levels(logits: torch.Tensor) -> torch.Tensor:
        """Convertit les K-1 probabilités cumulatives en niveau prédit (0 à K-1)."""
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)


class TubeFusionModel(nn.Module):
    def __init__(self, img_embedding_dim: int = 64, tabular_out_dim: int = 64, n_levels: int = N_LEVELS, pretrained: bool = False):
        super().__init__()
        self.cnn_encoder = SharedCNNEncoder(embedding_dim=img_embedding_dim, pretrained=pretrained)
        self.tabular_encoder = TabularEncoder(in_dim=1, out_dim=tabular_out_dim)

        # +1 : connexion directe (skip) du delta_T brut normalisé, en plus de son
        # embedding appris. Sans ça, le signal tabulaire (fort, quasi-discriminant
        # à lui seul) se fait diluer dans la fusion par les 2 branches image plus
        # volumineuses -> le modèle apprend à ignorer certaines classes (voir
        # matrice de confusion : niveau 1 collapsé sur 0, niveau 3 sur 4).
        fusion_dim = img_embedding_dim * 2 + tabular_out_dim + 1
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = OrdinalHead(in_dim=64, n_levels=n_levels)

    def forward(self, roi_test, roi_ref, delta_T):
        emb_test = self.cnn_encoder(roi_test)
        emb_ref = self.cnn_encoder(roi_ref)  # mêmes poids -> réseau siamois
        emb_tabular = self.tabular_encoder(delta_T)

        fused = torch.cat([emb_test, emb_ref, emb_tabular, delta_T], dim=1)  # delta_T brut inclus
        fused = self.fusion_mlp(fused)

        return self.head(fused)  # logits ordinaux (batch, n_levels - 1)


def coral_loss(logits: torch.Tensor, labels: torch.Tensor, n_levels: int = N_LEVELS) -> torch.Tensor:
    """Loss CORAL : transforme chaque label en vecteur binaire cumulatif
    (ex. label=2 sur 5 niveaux -> [1,1,0,0]) puis applique un BCE par seuil.
    """
    device = logits.device
    levels = torch.arange(n_levels - 1, device=device).unsqueeze(0)  # (1, n_levels-1)
    targets = (labels.unsqueeze(1) > levels).float()  # (batch, n_levels-1)
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)

