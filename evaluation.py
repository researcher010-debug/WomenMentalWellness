import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from architecture import TabMClassifier, DEVICE

DATA_DIR = Path("Data/preprocessed")
MODEL_DIR = Path("model")
DATASETS = ["women_depression", "maternal_depression"]
BATCH_SIZE = 512
GENERATE_SHAP = False
SHAP_BACKGROUND = 30
SHAP_SAMPLES = 50

def load_data(dataset_name):
    dataset_dir = DATA_DIR / dataset_name
    train_df = pd.read_csv(dataset_dir / "train.csv")
    test_df = pd.read_csv(dataset_dir / "test.csv")
    if "target" not in train_df.columns or "target" not in test_df.columns:
        raise ValueError(f"Target column missing in {dataset_name}.")
    feature_names = train_df.drop(columns=["target"]).columns.tolist()
    X_train = train_df[feature_names].to_numpy(dtype=np.float32)
    y_train = train_df["target"].to_numpy(dtype=np.int64)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.int64)
    if feature_names != test_df.drop(columns=["target"]).columns.tolist():
        raise ValueError(f"Feature columns do not match for {dataset_name}.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError(f"Non-finite values found in {dataset_name}.")
    return X_train, y_train, X_test, y_test, feature_names

def load_model(dataset_name):
    checkpoint_path = MODEL_DIR / dataset_name / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = TabMClassifier(input_dim=checkpoint["input_dim"], num_classes=checkpoint["num_classes"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint

def predict(model, X):
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.tensor(X[start:start + BATCH_SIZE], dtype=torch.float32, device=DEVICE)
            logits = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)

def calculate_metrics(y_true, y_pred, probabilities, classes):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred)
    }
    if len(classes) == 2:
        positive_class = classes[-1]
        binary_true = (y_true == positive_class).astype(int)
        positive_scores = probabilities[:, -1]
        if len(np.unique(binary_true)) == 2:
            metrics["roc_auc"] = auc(*roc_curve(binary_true, positive_scores)[:2])
            metrics["average_precision"] = average_precision_score(binary_true, positive_scores)
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
    else:
        binary_true = label_binarize(y_true, classes=classes)
        if binary_true.shape[1] == probabilities.shape[1] and all(len(np.unique(binary_true[:, i])) == 2 for i in range(len(classes))):
            metrics["roc_auc"] = float(np.mean([auc(*roc_curve(binary_true[:, i], probabilities[:, i])[:2]) for i in range(len(classes))]))
            metrics["average_precision"] = float(np.mean([average_precision_score(binary_true[:, i], probabilities[:, i]) for i in range(len(classes))]))
        else:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, output_dir, split_name):
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{split_name.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_confusion_matrix.png", dpi=300)
    plt.close()

def plot_roc_curve(y_true, probabilities, classes, output_dir, split_name):
    plt.figure(figsize=(7, 6))
    plotted = False
    if len(classes) == 2:
        binary_true = (y_true == classes[-1]).astype(int)
        if len(np.unique(binary_true)) == 2:
            fpr, tpr, _ = roc_curve(binary_true, probabilities[:, -1])
            plt.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.4f}")
            plotted = True
    else:
        binary_true = label_binarize(y_true, classes=classes)
        for index, class_label in enumerate(classes):
            if len(np.unique(binary_true[:, index])) == 2:
                fpr, tpr, _ = roc_curve(binary_true[:, index], probabilities[:, index])
                plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {auc(fpr, tpr):.4f})")
                plotted = True
    if plotted:
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{split_name.capitalize()} ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{split_name}_roc_curve.png", dpi=300)
    plt.close()

def plot_precision_recall(y_true, probabilities, classes, output_dir, split_name):
    plt.figure(figsize=(7, 6))
    plotted = False
    if len(classes) == 2:
        binary_true = (y_true == classes[-1]).astype(int)
        if len(np.unique(binary_true)) == 2:
            precision, recall, _ = precision_recall_curve(binary_true, probabilities[:, -1])
            score = average_precision_score(binary_true, probabilities[:, -1])
            plt.plot(recall, precision, label=f"AP = {score:.4f}")
            plotted = True
    else:
        binary_true = label_binarize(y_true, classes=classes)
        for index, class_label in enumerate(classes):
            if len(np.unique(binary_true[:, index])) == 2:
                precision, recall, _ = precision_recall_curve(binary_true[:, index], probabilities[:, index])
                score = average_precision_score(binary_true[:, index], probabilities[:, index])
                plt.plot(recall, precision, label=f"Class {class_label} (AP = {score:.4f})")
                plotted = True
    if plotted:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{split_name.capitalize()} Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{split_name}_pr_curve.png", dpi=300)
    plt.close()

def plot_training_history(dataset_name, output_dir):
    history_path = output_dir / "history.json"
    if not history_path.exists():
        warnings.warn(f"Training history not found for {dataset_name}.")
        return
    with open(history_path, "r") as file:
        history = json.load(file)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name} Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
    plt.close()

def generate_shap_explanations(model, X_train, X_test, feature_names, output_dir):
    try:
        import shap
    except ImportError:
        warnings.warn("SHAP is not installed. Skipping SHAP explanations.")
        return
    rng = np.random.RandomState(42)
    background_indices = rng.choice(len(X_train), min(SHAP_BACKGROUND, len(X_train)), replace=False)
    sample_indices = rng.choice(len(X_test), min(SHAP_SAMPLES, len(X_test)), replace=False)
    background = X_train[background_indices]
    samples = X_test[sample_indices]
    def predict_probability(data):
        return predict(model, np.asarray(data, dtype=np.float32))
    explainer = shap.KernelExplainer(predict_probability, background)
    shap_values = explainer.shap_values(samples, nsamples=100)
    if isinstance(shap_values, list):
        for class_index, values in enumerate(shap_values):
            plt.figure()
            shap.summary_plot(values, samples, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_summary_class_{class_index}.png", dpi=300, bbox_inches="tight")
            plt.close()
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            for class_index in range(values.shape[-1]):
                plt.figure()
                shap.summary_plot(values[:, :, class_index], samples, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(output_dir / f"shap_summary_class_{class_index}.png", dpi=300, bbox_inches="tight")
                plt.close()
        else:
            plt.figure()
            shap.summary_plot(values, samples, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
            plt.close()

def evaluate_split(model, X, y, classes, output_dir, split_name):
    probabilities = predict(model, X)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = calculate_metrics(y, predictions, probabilities, classes)
    metrics["classification_report"] = classification_report(y, predictions, labels=classes, zero_division=0, output_dict=True)
    with open(output_dir / f"{split_name}_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)
    plot_confusion_matrix(y, predictions, classes, output_dir, split_name)
    plot_roc_curve(y, probabilities, classes, output_dir, split_name)
    plot_precision_recall(y, probabilities, classes, output_dir, split_name)
    return {key: value for key, value in metrics.items() if key != "classification_report"}

def evaluate_dataset(dataset_name):
    output_dir = MODEL_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test, feature_names = load_data(dataset_name)
    model, checkpoint = load_model(dataset_name)
    classes = np.asarray(checkpoint["classes"], dtype=np.int64)
    if X_train.shape[1] != checkpoint["input_dim"]:
        raise ValueError(f"Input dimension mismatch for {dataset_name}.")
    train_metrics = evaluate_split(model, X_train, y_train, classes, output_dir, "train")
    test_metrics = evaluate_split(model, X_test, y_test, classes, output_dir, "test")
    plot_training_history(dataset_name, output_dir)
    if GENERATE_SHAP:
        generate_shap_explanations(model, X_train, X_test, feature_names, output_dir)
    results = []
    for split_name, metrics in [("train", train_metrics), ("test", test_metrics)]:
        results.append({"dataset": dataset_name, "split": split_name, **metrics})
    pd.DataFrame(results).to_csv(output_dir / "evaluation_summary.csv", index=False)
    print(f"\n{dataset_name.upper()} EVALUATION")
    print(pd.DataFrame(results).to_string(index=False))
    return results

def main():
    all_results = []
    for dataset_name in DATASETS:
        all_results.extend(evaluate_dataset(dataset_name))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(MODEL_DIR / "evaluation_summary.csv", index=False)

if __name__ == "__main__":
    main()