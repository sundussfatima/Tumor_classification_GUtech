import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns

def plot_history(history):
    """
    Plots training/validation accuracy and loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.show()


def evaluate_model(model, history, val_ds):
    """
    Evaluates model, prints report and plots results.
    """
    plot_history(history)

    y_true = val_ds.classes
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(val_ds.class_indices.keys())))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=val_ds.class_indices.keys(),
                yticklabels=val_ds.class_indices.keys())
    plt.title("Confusion Matrix")
    plt.show()
