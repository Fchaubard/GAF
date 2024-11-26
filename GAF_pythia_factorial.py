# THIS SCRIPT TRAINS A PYTHIA MODEL WITH OR WITHOUT COT FACTORIAL DATA and GAF

import torch
import pdb
import torch.nn as nn
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import numpy as np
import math
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import numpy as np
import random
import wandb
import os
import argparse



os.environ["WANDB_API_KEY"] = ""

config = {}



####################################################################################################
# Configuration and Data Preparation
####################################################################################################
verbose = True
total_iterations = 1000000
num_iters_per_validation = 5

config['tau'] = 0.95
devvv = 5 # set device to 0, 2, 4, 6, or 8

config['train_numbers'] = list(range(1, 10)) + list(range(15, 20))
config['test_numbers'] = list(range(10, 15)) + list(range(20, 25))
config['val']=True

config['warmup_steps'] = 100
config['lr'] = 1e-6
config['betas'] = (.9,.99)
config['weight_decay'] = 0.05
config['with_COT'] = False


device = torch.device('cuda:'+str(devvv) if torch.cuda.is_available() else 'cpu')
device_ids = [devvv, devvv+1]  # GPUs cuda:0 and cuda:1

print(f'Using device: {device}')


# Initialize WandB
wandb.init(project="Pythia+Factorial", config=config)


def preprocess_for_generation(input_ids, attention_mask):
    # Find the position of the <sep> token
    sep_token_id = tokenizer.convert_tokens_to_ids('<sep>')
    sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
    
    if len(sep_positions) > 0:
        # Cut off everything after the first <sep> token (inclusive)
        cut_position = sep_positions[0] + 1  # Include the <sep> token
        input_ids = input_ids[:cut_position]
        attention_mask = attention_mask[:cut_position]
    else:
        # If <sep> not found, use the whole input_ids
        print("heree!!!")
        pass  # Handle accordingly if necessary
    
    return input_ids, attention_mask

def format_sample_no_COT(n):
    significant_figures = format_decimal(compute_factorial(n))
    sample = f"<bos> {n}! <sep> #### {significant_figures} <eos>"
    return sample

def format_sample_with_COT(n):
    from decimal import Decimal, getcontext
    getcontext().prec = 1000  # High precision for large factorials
    steps = []
    current_total = Decimal(n)
    for i in range(n - 1, 0, -1):
        next_total = current_total * Decimal(i)
        # Format numbers with 7 significant figures
        current_total_str = format_decimal(current_total)
        next_total_str = format_decimal(next_total)
        # Include the <multiply> token and the result
        step_str = f"{current_total_str}*{i}=<multiply>{next_total_str}"
        steps.append(step_str)
        current_total = next_total
    steps_str = " <sep> ".join(steps)
    final_answer = format_decimal(current_total)
    sample = f"<bos> {n}! <sep> {steps_str} <sep> so the answer is {final_answer}. #### {final_answer} <eos>"
    return sample

def format_decimal(num):
    return f"{float(num):.7g}"

format_sample = format_sample_no_COT # format_sample_with_COT # format_sample_no_COT

# Set random seed for reproducibility
random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-410m')
model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/pythia-410m')

# Add special tokens to the tokenizer, including '<multiply>'
tokenizer.add_special_tokens({
    'pad_token': '<pad>',
    'bos_token': '<bos>',
    'sep_token': '<sep>',
    'eos_token': '<eos>',
    'additional_special_tokens': ['<multiply>']
})

# Update the model's embeddings to accommodate the new tokens
model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# Wrap the model with DataParallel to use multiple GPUs

model = torch.nn.DataParallel(model, device_ids=device_ids)
model.to(device)

# Prepare the dataset
def compute_factorial(n):
    return math.factorial(n)

# Generate training and test data
train_samples = [format_sample(n) for n in config['train_numbers']]
test_samples = [format_sample(n) for n in config['test_numbers']]

def calculate_per_token_accuracy(test_logits, test_labels):
    shift_logits = test_logits[..., :-1, :].contiguous()
    shift_predictions = shift_logits.argmax(dim=-1)
    shift_labels = test_labels[..., 1:].contiguous()
    shift_mask = shift_labels != tokenizer.pad_token_id
    shift_correct = (shift_predictions == shift_labels) & shift_mask
    per_token_accuracy = shift_correct.sum().item() / shift_mask.sum().item()
    return per_token_accuracy

# Tokenize the data
def tokenize_samples(samples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for sample in samples:
        # Tokenize the sample with offsets mapping
        encoded = tokenizer.encode_plus(
            sample,
            return_tensors='pt',
            padding=False,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]
        offsets = encoded['offset_mapping'][0]
        
        # Initialize labels as the same as input_ids
        labels = input_ids.clone()
        
        # Get the text
        text = sample
        
        # Find the positions of '<multiply>' in the tokenized input_ids
        multiply_token_id = tokenizer.convert_tokens_to_ids('<multiply>')
        multiply_positions = (input_ids == multiply_token_id).nonzero(as_tuple=True)[0]
        
        for pos in multiply_positions:
            # The character position corresponding to this token
            start_char = offsets[pos][0].item()
            end_char = offsets[pos][1].item()
            # Find the multiplication result immediately after '<multiply>'
            match = re.search(r'\s*([\d\.e\+\-]+)', text[end_char:])
            if match:
                number_start = end_char + match.start()
                number_end = end_char + match.end()
                # Find the tokens corresponding to this range
                token_positions = []
                for idx, (token_start, token_end) in enumerate(offsets):
                    token_start = token_start.item()
                    token_end = token_end.item()
                    if token_start >= number_end:
                        break
                    if token_end > number_start and token_start < number_end:
                        token_positions.append(idx)
                # Set labels at these positions to -100
                labels[token_positions] = -100
        # Append to the lists
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    # Pad the sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )
    return {'input_ids': input_ids_padded, 'attention_mask': attention_mask_padded, 'labels': labels_padded}

train_data = tokenize_samples(train_samples)
test_data = tokenize_samples(test_samples)

# Move data to device
for key in train_data:
    train_data[key] = train_data[key].to(device)
for key in test_data:
    test_data[key] = test_data[key].to(device)

# Set up optimizer and loss function
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)



# Initialize metrics storage
train_losses = []
test_losses = []
train_per_token_accuracies = []
test_per_token_accuracies = []
train_final_answer_accuracies = []
test_final_answer_accuracies = []
train_iterations = []
test_iterations = []


def filter_gradients(G1, G2, cos_distance_thresh=1):
    # Flatten G1 and G2 into vectors
    G1_flat = torch.cat([g1.view(-1) for g1 in G1])
    G2_flat = torch.cat([g2.view(-1) for g2 in G2])

    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(G1_flat, G2_flat, dim=0)

    # Compute cosine distance
    cos_distance = 1 - cos_sim

    # If cos_distance > cos_distance_thresh, filtered_grad == None
    if cos_distance > cos_distance_thresh:
        filtered_grad = None
    else:
        # Compute average of G1 and G2
        filtered_grad = [(g1 + g2) / 2 for g1, g2 in zip(G1, G2)]

    return filtered_grad, cos_distance.item()


# Optimized generate_with_calculator function
def generate_with_calculator(model, tokenizer, input_ids, attention_mask, max_length, device):
    with torch.no_grad():
        generated_ids = input_ids.clone().to(device)
        multiply_token_id = tokenizer.convert_tokens_to_ids('<multiply>')
        eos_token_id = tokenizer.eos_token_id
        sep_token_id = tokenizer.convert_tokens_to_ids('<sep>')

        # Process the initial input_ids to get past_key_values
        outputs = model(
            input_ids=input_ids.to(device),
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        last_operator_pos = generated_ids.shape[1]

        for _ in range(max_length):
            # Generate next token
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            # Append the next token
            generated_ids = torch.cat((generated_ids, next_token_id.unsqueeze(0)), dim=1)

            # If the next token is '<multiply>', compute multiplication
            if next_token_id.item() == multiply_token_id:
                tokens_to_decode = generated_ids[0, last_operator_pos:]
                text_to_parse = tokenizer.decode(tokens_to_decode, skip_special_tokens=False)
                # Extract operands immediately before '<multiply>'
                match = re.search(r'([\d\.e\+\-]+)\s*\*\s*([\d\.e\+\-]+)\s*=\s*<multiply>$', text_to_parse)
                if match:
                    operand1 = float(match.group(1))
                    operand2 = float(match.group(2))
                    result = operand1 * operand2
                    result_str = format_decimal(result)
                    # Tokenize and append the result
                    result_tokens = tokenizer.encode(result_str, return_tensors='pt').to(device)
                    generated_ids = torch.cat((generated_ids, result_tokens), dim=1)
                    # Update past_key_values
                    outputs = model(
                        input_ids=result_tokens.to(device),
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.past_key_values
                    last_operator_pos = generated_ids.shape[1]
                    next_token_logits = outputs.logits[:, -1, :]
                    continue  # Proceed to next iteration
            elif next_token_id.item() == sep_token_id:
                last_operator_pos = generated_ids.shape[1]

            # Check for end of sequence
            if next_token_id.item() == eos_token_id:
                break

            # Get next token logits
            outputs = model(
                input_ids=next_token_id.unsqueeze(0).to(device),
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
    return generated_ids



optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=config['betas'],  weight_decay=config['weight_decay'])
criterion = nn.CrossEntropyLoss()


# Learning Rate Scheduler with Warmup
warmup_steps = config['warmup_steps']
total_steps = total_iterations
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps), eta_min=0)


# Training loop with gradient filtering
for iteration in tqdm(range(0, total_iterations + 1)):

    if iteration >= 0: # just in case you want to skip the first one for logging val
        if iteration < warmup_steps:
            lr = config['lr'] * (iteration / warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()  # Adjust learning rate according to cosine schedule

        print("Performing Train")
        torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad()

        # Sample two random mini-batches
        n_samples = len(train_samples)
        batch_size = max(1, n_samples // 2)

        indices1 =  [i for i in range( int(n_samples/2))]
        indices2 =  [i for i in range(int(n_samples/2),n_samples)]

        batch1 = {key: train_data[key][indices1] for key in train_data}
        batch2 = {key: train_data[key][indices2] for key in train_data}

        # First mini-batch
        outputs1 = model(**batch1)
        loss1 = outputs1.loss.mean()
        loss1.backward()
        G1 = [p.grad.clone() for p in model.parameters()]
        optimizer.zero_grad()

        # Second mini-batch
        outputs2 = model(**batch2)
        loss2 = outputs2.loss.mean()
        loss2.backward()
        G2 = [p.grad.clone() for p in model.parameters()]

        # Filter gradients
        filtered_grad, cosine_distance = filter_gradients(G1, G2, config['tau'])

        if not filtered_grad == None:
            # Apply filtered gradients
            with torch.no_grad():
                for param, grad in zip(model.parameters(), filtered_grad):
                    param.grad = grad

            optimizer.step()
            # Store train loss
            train_loss = (loss1.item() + loss2.item()) / 2
            train_losses.append(train_loss)
    
            # Compute per-token train accuracy
            with torch.no_grad():
                per_token_accuracy1 = calculate_per_token_accuracy(outputs1.logits, batch1['labels'])
                per_token_accuracy2 = calculate_per_token_accuracy(outputs2.logits, batch2['labels'])
                per_token_accuracy = (per_token_accuracy1 + per_token_accuracy2) / 2
                train_loss = (loss1 + loss2) / 2
                train_per_token_accuracies.append(per_token_accuracy)
                train_iterations.append(iteration)

        else:
            print(f"Skipping update due to cosine_distance = {cosine_distance}") 
    
    # Validation and plotting
    if iteration % num_iters_per_validation == 0 and config['val']==True:
        print("Performing Val")
        test_iterations.append(iteration)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(**test_data)
            test_loss = test_outputs.loss.mean().item()
            test_losses.append(test_loss)
            test_per_token_accuracy = calculate_per_token_accuracy(test_outputs.logits, test_data['labels'])
            test_per_token_accuracies.append(test_per_token_accuracy)


        print(f"generating on train: iter {iteration}")
        # Compute final answer accuracies
        # For train data
        train_final_correct = 0
        listt = list(range(len(train_samples)))
        for i in listt:#[:3]:
            # Original input_ids and attention_mask
            input_ids = train_data['input_ids'][i]
            attention_mask = train_data['attention_mask'][i]
            
            # Preprocess to exclude the answer
            input_ids, attention_mask = preprocess_for_generation(input_ids, attention_mask)
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            generated = generate_with_calculator(
                model.module,
                tokenizer,
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=train_data['input_ids'][i].shape[0],
                device=device
            )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
            if verbose and iteration>1000:
                # Extract the answer
                print("="*50)
                print("TRAIN:")
                print(f"train truth: {train_samples[i]}")
                print(f"generated_text: {generated_text}")
                time.sleep(0.1)
            match = re.search(r'####\s*([\d\.e\+\-]+)', generated_text)
            if match:
                generated_answer = match.group(1)
                target_answer = re.search(r'####\s*([\d\.e\+\-]+)', train_samples[i]).group(1)
                if generated_answer.strip() == target_answer.strip():
                    train_final_correct += 1
        train_final_accuracy = train_final_correct / len(train_samples)
        train_final_answer_accuracies.append(train_final_accuracy)
        print(f"testing: iter {iteration}")
        # For test data
        test_final_correct = 0
        listt = list(range(len(test_samples)))
        for i in listt:#[:3]:
            # Original input_ids and attention_mask
            input_ids = test_data['input_ids'][i]
            attention_mask = test_data['attention_mask'][i]
            
            # Preprocess to exclude the answer
            input_ids, attention_mask = preprocess_for_generation(input_ids, attention_mask)

            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            
            generated = generate_with_calculator(
                model.module,
                tokenizer,
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_length=train_data['input_ids'][i].shape[0],
                device=device
            )
            
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
            if verbose: # and iteration>1000:
                print("="*50)
                print("TEST:")
                print(f"test truth: {test_samples[i]}")
                print(f"generated_text: {generated_text}")
                time.sleep(0.1)
                
            # Extract the answer
            match = re.search(r'####\s*([\d\.e\+\-]+)', generated_text)
            
            if match:
                generated_answer = match.group(1)
                target_answer = re.search(r'####\s*([\d\.e\+\-]+)', test_samples[i]).group(1)
                if generated_answer.strip() == target_answer.strip():
                    test_final_correct += 1
        test_final_accuracy = test_final_correct / len(test_samples)
        test_final_answer_accuracies.append(test_final_accuracy)



                ##### Log metrics to wandb
        try:
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy_per_tok': per_token_accuracy,
                'train_final_accuracy': train_final_accuracy,
                'val_loss': test_loss,
                'val_accuracy_per_tok': test_per_token_accuracy,
                'val_final_accuracy': test_final_accuracy,
                'cosine_distance': cosine_distance,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'iteration': iteration,
            })
        except Exception as e:
            print(f"Failed to log to wandb: {e}")

        
        # Print metrics
        print(f"Iteration {iteration}")
        if iteration > 0:
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Per Token Train Accuracy: {per_token_accuracy*100:.2f}%")
            print(f"Per Token Test Accuracy: {test_per_token_accuracy*100:.2f}%")
            print(f"Final Answer Train Accuracy: {train_final_accuracy*100:.2f}%")
            print(f"Final Answer Test Accuracy: {test_final_accuracy*100:.2f}%")
        

        
        # # Plot the metrics
        # plt.figure(figsize=(12, 8))
        # plt.subplot(2, 2, 1)
        # plt.plot(train_iterations, train_losses, label='Train Loss')
        # plt.plot(test_iterations, test_losses, label='Test Loss')
        # plt.xlabel('Iterations')
        # plt.ylabel('Loss')
        # plt.title('Loss')
        # plt.legend()
        
        # plt.subplot(2, 2, 2)
        # plt.plot(train_iterations, train_per_token_accuracies, label='Train Per Token Accuracy')
        # plt.plot(test_iterations, test_per_token_accuracies, label='Test Per Token Accuracy')
        # plt.xlabel('Iterations')
        # plt.ylabel('Accuracy')
        # plt.title('Per Token Accuracy')
        # plt.legend()

                
        # plt.subplot(2, 2, 3)
        # plt.plot(test_iterations, train_final_answer_accuracies, label='Train Final Answer Accuracy')
        # plt.plot(test_iterations, test_final_answer_accuracies, label='Test Final Answer Accuracy')
        # plt.xlabel('Iterations')
        # plt.ylabel('Accuracy')
        # plt.title('Final Answer Acc')
        # plt.legend()
        
        # plt.tight_layout()
        # plt.show()
