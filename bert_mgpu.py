import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import time
import math
import os
import argparse
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader, DistributedSampler
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup_ddp():
    dist.destroy_process_group()

def train_cross_encoder(rank, world_size, csv_filename, precision="default", epochs=5, batch_size=32, learning_rate=2e-5, model_folder_name="cross_encoder_model"):
    """Trains a CrossEncoder model for sentence pair classification using a dataset from CSV with DDP."""
    
    setup_ddp()
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
    
    precision_modes = ["default", "mixed", "fp16", "bf16", "tf32"] if precision == "all" else [precision]
    
    for prec in precision_modes:
        print(f"Starting training with precision: {prec}")
        
        model = CrossEncoder('google-bert/bert-base-multilingual-uncased', num_labels=1)
        device = torch.device(f"cuda:{rank}")
        model.model.to(device)
        model.model = torch.nn.parallel.DistributedDataParallel(model.model, device_ids=[rank])
        
        # Set precision mode
        if prec == "mixed":
            model.model.to(torch.bfloat16)
        elif prec == "fp16":
            model.model.to(torch.float16)
        elif prec == "bf16":
            model.model.to(torch.bfloat16)
        elif prec == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Using TF32 precision")
        
        # Print training precision
        param_dtype = next(model.model.parameters()).dtype
        print(f"Training is happening in precision: {param_dtype}")
        
        train_examples = [InputExample(texts=[sent1, sent2], label=label) for (sent1, sent2), label in zip(X_train, y_train)]
        val_examples = [InputExample(texts=[sent1, sent2], label=label) for (sent1, sent2), label in zip(X_val, y_val)]
        
        train_dataloader = DataLoader(train_examples, sampler=DistributedSampler(train_examples), batch_size=batch_size)
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
                output_path=(model_folder_name + f'_{prec}_best'),
            )
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            
            eval_score = evaluator(model)
            print(f"Epoch {epoch+1} - Accuracy: {eval_score:.4f}")
        
        training_time = time.time() - start_time
        model.save(model_folder_name + f'_{prec}')
        print(f"Model saved at: {model_folder_name}_{prec}")
        print(f"Training Time: {training_time:.2f} seconds")
    
    cleanup_ddp()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CrossEncoder model using a CSV dataset with DDP.",
                                     epilog="Example usage: python -m torch.distributed.launch --nproc_per_node=4 script.py --csv dataset.csv --train fp16")
    parser.add_argument("--train", type=str, choices=["default", "mixed", "fp16", "bf16", "tf32", "all"], default="default", help="Precision mode for training")
    parser.add_argument("--csv", type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(train_cross_encoder, args=(world_size, args.csv, args.train), nprocs=world_size, join=True)
