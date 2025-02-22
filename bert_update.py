import torch
import pandas as pd
import time
import math
import os
import argparse
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

# Set GPU if available
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_cross_encoder(csv_filename, precision="default", epochs=5, batch_size=32, learning_rate=2e-5, model_folder_name="cross_encoder_model"):
    """Trains a CrossEncoder model for sentence pair classification using a dataset from CSV."""
    
    df = pd.read_csv(csv_filename)
    
    # Data Cleaning: Remove NaN values and empty strings
    df.dropna(inplace=True)
    df = df[df.iloc[:, 0].str.strip() != ""]
    df = df[df.iloc[:, 1].str.strip() != ""]
    
    X_train = list(zip(df.iloc[:, 0], df.iloc[:, 1]))  # First column: English, Second column: French
    y_train = [float(1.0)] * len(X_train)  # Assuming all translations are correct (label=1)
    
    split_idx = int(0.9 * len(df))  # 90% training, 10% validation
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]
    
    model = CrossEncoder('google-bert/bert-base-multilingual-uncased', num_labels=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)
    
    # Set precision mode
    if precision == "mixed":
        model.model.to(torch.bfloat16)
    elif precision == "fp16":
        model.model.to(torch.float16)
    elif precision == "bf16":
        model.model.to(torch.bfloat16)
    elif precision == "int8":
        model.model = torch.quantization.quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8)
    elif precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Using TF32 precision")
    
    # Print training precision
    param_dtype = next(model.model.parameters()).dtype
    print(f"Training is happening in precision: {param_dtype}")
    
    train_examples = [InputExample(texts=[sent1, sent2], label=label) for (sent1, sent2), label in zip(X_train, y_train)]
    val_examples = [InputExample(texts=[sent1, sent2], label=label) for (sent1, sent2), label in zip(X_val, y_val)]
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name="validation")
    
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            save_best_model=True,
            output_path=(model_folder_name + '_best'),
        )
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        
        eval_score = evaluator(model)
        print(f"Epoch {epoch+1} - Accuracy: {eval_score:.4f}")
    
    training_time = time.time() - start_time
    model.save(model_folder_name)
    print(f"Model saved at: {model_folder_name}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CrossEncoder model using a CSV dataset.",
                                     epilog="Example usage: python script.py --csv dataset.csv --train fp16")
    parser.add_argument("--train", type=str, choices=["default", "mixed", "fp16", "bf16", "int8", "tf32"], default="default", help="Precision mode for training")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()
    
    trained_model = train_cross_encoder(args.csv, precision=args.train)
