import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from architecture import TabMClassifier, reset_random, DEVICE

SEED = 42
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_SIZE = 0.15
PATIENCE = 20
DATA_DIR = Path("Data/preprocessed")
MODEL_DIR = Path("model")
DATASETS = ["women_depression", "maternal_depression"]

class Adan(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8, weight_decay=1e-4):
        if lr <= 0 or eps <= 0:
            raise ValueError("Learning rate and epsilon must be positive.")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients.")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_diff"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["previous_grad"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_diff = state["exp_avg_diff"]
                exp_avg_sq = state["exp_avg_sq"]
                previous_grad = state["previous_grad"]
                grad_diff = grad - previous_grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_diff.mul_(beta2).add_(grad_diff, alpha=1 - beta2)
                grad_nesterov = grad + beta2 * grad_diff
                exp_avg_sq.mul_(beta3).addcmul_(grad_nesterov, grad_nesterov, value=1 - beta3)
                previous_grad.copy_(grad)
                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                bias3 = 1 - beta3 ** step
                numerator = exp_avg / bias1 + beta2 * exp_avg_diff / bias2
                denominator = (exp_avg_sq / bias3).sqrt().add_(group["eps"])
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(numerator, denominator, value=-group["lr"])
        return loss

def load_dataset(dataset_name):
    dataset_dir = DATA_DIR / dataset_name
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing preprocessed files for {dataset_name}: {train_path} or {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "target" not in train_df.columns or "target" not in test_df.columns:
        raise ValueError(f"'target' column is missing in {dataset_name}.")
    X_train = train_df.drop(columns=["target"]).astype(np.float32).values
    y_train = train_df["target"].astype(np.int64).values
    X_test = test_df.drop(columns=["target"]).astype(np.float32).values
    y_test = test_df["target"].astype(np.int64).values
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Training and testing feature dimensions do not match for {dataset_name}.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError(f"Non-finite feature values found in {dataset_name}.")
    return X_train, y_train, X_test, y_test

def create_loader(X, y, shuffle=False):
    features = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(features, labels), batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=torch.cuda.is_available())

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

def train_dataset(dataset_name):
    reset_random(SEED)
    X, y, X_test, y_test = load_dataset(dataset_name)
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(f"{dataset_name} must contain at least two classes.")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SIZE, random_state=SEED, stratify=y)
    train_loader = create_loader(X_train, y_train, shuffle=True)
    val_loader = create_loader(X_val, y_val)
    test_loader = create_loader(X_test, y_test)
    model = TabMClassifier(input_dim=X.shape[1], num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adan(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6)
    save_dir = MODEL_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "learning_rate": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)
        print(f"{dataset_name} | Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "input_dim": X.shape[1], "num_classes": len(classes), "classes": classes.tolist(), "best_epoch": best_epoch, "validation_loss": best_val_loss}, save_dir / "best_model.pth")
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break
    with open(save_dir / "history.json", "w") as file:
        json.dump(history, file, indent=4)
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)
    checkpoint = torch.load(save_dir / "best_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion)
    results = {"dataset": dataset_name, "best_epoch": best_epoch, "best_validation_loss": best_val_loss, "test_loss": test_loss, "test_accuracy": test_accuracy}
    with open(save_dir / "training_results.json", "w") as file:
        json.dump(results, file, indent=4)
    print(f"{dataset_name} | Best Epoch: {best_epoch} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    return results

def main():
    results = []
    for dataset_name in DATASETS:
        results.append(train_dataset(dataset_name))
    pd.DataFrame(results).to_csv(MODEL_DIR / "training_summary.csv", index=False)

if __name__ == "__main__":
    main()