# -*- coding: utf-8 -*-
"""graphs.py

"""

import matplotlib.pyplot as plt

# Extracted data from logs (first 30 epochs)
epochs = list(range(1, 61))
train_accuracy #get from results  


val_accuracy #get from results  


train_loss #get from results  


val_loss #get from results  


learning_rate #get from results  


best_val_acc #get from results  


# Plotting
plt.figure(figsize=(15, 12))

# Training and Validation Accuracy
plt.subplot(3, 2, 1)
plt.plot(epochs, train_accuracy, label='Train Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epoch")
plt.legend()

# Training and Validation Loss
plt.subplot(3, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs. Epoch")
plt.legend()

# Learning Rate
plt.subplot(3, 2, 3)
plt.plot(epochs, learning_rate, label='Learning Rate')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs. Epoch")
plt.legend()

# Validation Accuracy Improvement
plt.subplot(3, 2, 4)
plt.plot(epochs, best_val_acc, label='Best Validation Accuracy So Far')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Improvement")
plt.legend()

# Only Training Accuracy
plt.subplot(3, 2, 5)
plt.plot(epochs, train_accuracy, color='blue')
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy vs. Epoch")

# Only Validation Accuracy
plt.subplot(3, 2, 6)
plt.plot(epochs, val_accuracy, color='orange')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs. Epoch")

plt.tight_layout()
plt.show()

# 1. Generalization Gap vs. Epochs
gen_gap_acc = [tr - val for tr, val in zip(train_accuracy, val_accuracy)]
plt.figure(figsize=(6, 4))
plt.plot(epochs, gen_gap_acc, color='purple')
plt.title('Generalization Gap (Accuracy) vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Train Acc - Val Acc')
plt.tight_layout()
plt.show()

# 2. Validation Accuracy Improvement vs. Epochs
plt.figure(figsize=(6, 4))
plt.plot(epochs, best_val_acc, color='green')
plt.title('Best-so-far Validation Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Best Validation Accuracy')
plt.tight_layout()
plt.show()

# 3. Loss Comparison Bar Plot
plt.figure(figsize=(5, 4))
plt.bar(['Training Loss', 'Validation Loss'], [train_loss[-1], val_loss[-1]], color=['blue', 'orange'])
plt.title('Final Loss Comparison')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# 4. Accuracy Comparison Bar Plot
plt.figure(figsize=(5, 4))
plt.bar(['Training Accuracy', 'Validation Accuracy'], [train_accuracy[-1], val_accuracy[-1]], color=['blue', 'orange'])
plt.title('Final Accuracy Comparison')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()


import pandas as pd


df = pd.read_excel('results.xlsx') 

train_accuracy = df['Train Accuracy'].tolist()
print(train_accuracy)

val_accuracy = df['Validation Accuracy'].tolist()
print(val_accuracy)

train_loss = df['Train Loss'].tolist()
print(train_loss)

val_loss = df['Validation Loss'].tolist()
print(val_loss)

learning_rate = df['Learning Rate'].tolist()
print(learning_rate)

best_val_acc = df['best validation acc so far'].tolist()
print(best_val_acc)

import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = list(range(1, 61))

train_acc #get from results  


val_acc #get from results  


train_loss #get from results  


val_loss #get from results  


# --- 1. Accuracy Difference Trend (Train Acc - Val Acc) ---
acc_diff = [tr - val for tr, val in zip(train_acc, val_acc)]

plt.figure(figsize=(7, 4))
plt.plot(epochs, acc_diff,  linestyle='-', color='red')
plt.title('Accuracy Difference Trend (Train - Validation)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy Gap')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 2. Loss Plateau Detection (Loss Delta Plot) ---
# Calculate the absolute change in validation loss
val_loss_change = [0] + [abs(val_loss[i] - val_loss[i-1]) for i in range(1, len(val_loss))]

plt.figure(figsize=(7, 4))
plt.plot(epochs, val_loss, marker='o', label='Validation Loss', color='orange')
plt.plot(epochs, val_loss_change, marker='x', label='Loss Change (Δ)', linestyle='--', color='blue')
plt.title('Loss Plateau Detection')
plt.xlabel('Epochs')
plt.ylabel('Loss / Loss Change')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()