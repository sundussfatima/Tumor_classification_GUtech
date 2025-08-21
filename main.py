from preprocessing import load_data
from model_dev import build_and_train_model
from results import evaluate_model

if __name__ == "__main__":
    # Step 1: Load Data
    train_ds, val_ds = load_data(data_dir="dataset")

    # Step 2: Build and Train Model
    model, history = build_and_train_model(train_ds, val_ds, epochs=30)

    # Step 3: Evaluate Results
    evaluate_model(model, history, val_ds)
