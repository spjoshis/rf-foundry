import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from models.cnn_extractor import PriceCNN

class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)

        self.price_cnn = PriceCNN()
        n_indicators = observation_space["indicators"].shape[0]

        self.indicator_mlp = nn.Sequential(
            nn.Linear(n_indicators, 64),
            nn.ReLU()
        )

    def forward(self, obs):
        price_emb = self.price_cnn(obs["price"])
        ind_emb = self.indicator_mlp(obs["indicators"])
        return torch.cat([price_emb, ind_emb], dim=1)
