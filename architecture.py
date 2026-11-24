import random
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reset_random(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TabMBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_models, dropout=0.2):
        super().__init__()
        self.shared_layer = nn.Linear(input_dim, output_dim)
        self.model_scale = nn.Parameter(torch.ones(1, num_models, output_dim))
        self.model_bias = nn.Parameter(torch.zeros(1, num_models, output_dim))
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.shared_layer(x)
        x = x * self.model_scale + self.model_bias
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)

class TabMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128), num_models=8, dropout=0.2):
        super().__init__()
        if input_dim < 1 or num_classes < 2 or num_models < 1:
            raise ValueError("Invalid input_dim, num_classes, or num_models.")
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_models = num_models
        self.input_norm = nn.LayerNorm(input_dim)
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(TabMBlock(previous_dim, hidden_dim, num_models, dropout))
            previous_dim = hidden_dim
        self.backbone = nn.ModuleList(layers)
        self.classifier = nn.Linear(previous_dim, num_classes)
        self.model_bias = nn.Parameter(torch.zeros(1, num_models, num_classes))
        self.to(DEVICE)

    def forward(self, x, return_submodel_logits=False):
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input shape (batch_size, {self.input_dim}).")
        x = self.input_norm(x.float()).unsqueeze(1).expand(-1, self.num_models, -1)
        for layer in self.backbone:
            x = layer(x)
        submodel_logits = self.classifier(x) + self.model_bias
        logits = submodel_logits.mean(dim=1)
        if return_submodel_logits:
            return logits, submodel_logits
        return logits

def build_model(input_dim, num_classes, hidden_dims=(256, 128), num_models=8, dropout=0.2):
    return TabMClassifier(input_dim, num_classes, hidden_dims, num_models, dropout).to(DEVICE)