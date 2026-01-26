"""## Import necessary libraries and modules"""

import pickle, numpy as np, scipy as sp, torch, random, os, json, sys, logging, argparse, time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datasets import load_dataset
from scipy.special import softmax
from scipy.stats import entropy

"""## Define functions for processing"""

def load_all_chunks(**load_chunk_args):
    """
    Load the specific data corresponding to the `data_key` from chunked pickle files.
    
    Args:
        data_key (str): The key for the data we want to extract (e.g., 'logits', 'correct', etc.).
        chunk_dir (str): The directory where the chunk files are stored.
    
    Returns:
        np.ndarray: An array containing the data for the given `data_key`.
    """
    dataset_name = load_chunk_args.get('dataset_name')
    data_category = load_chunk_args.get('data_category')
    chunk_dir = load_chunk_args.get('chunk_dir')
    data_key = load_chunk_args.get('data_key')

    chunk_data = []
    chunk_idx = 1
    
    if dataset_name == "mTREx":
        while chunk_idx <= load_chunk_args.get('max_num_samples'):
            chunk_file_path = os.path.join(chunk_dir, f"{data_category}_chunk_{chunk_idx}.pickle")
            if not os.path.exists(chunk_file_path):
                break
            
            with open(chunk_file_path, "rb") as chunk_file:
                # Load each entry in the chunk and extract the relevant data
                while True:
                    try:
                        result_entry = pickle.load(chunk_file)
                        if data_key in result_entry:
                            chunk_data.append(result_entry[data_key])  # Add the data for the given key
                    except EOFError:
                        break
            chunk_idx += 1

    elif dataset_name == "GMMLU":
        subjects = load_chunk_args.get('gmmlu_subjects')
        for subject in subjects:
            # Load chunk file which have names like loop through all chunk files
            chunk_file_path = os.path.join(chunk_dir, f"{data_category}_{subject}_chunk.pickle")
            
            with open(chunk_file_path, "rb") as chunk_file:
                # Load each entry in the chunk and extract the relevant data
                while True:
                    try:
                        result_entry = pickle.load(chunk_file)
                        if data_key in result_entry:
                            chunk_data.append(result_entry[data_key])  # Add the data for the given key
                    except EOFError:
                        break
            chunk_idx += 1

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Supported datasets are 'mTREx' and 'GMMLU'.")
    
    return chunk_data

# Define the classifier classes
class FFHallucinationClassifier(torch.nn.Module):
    def __init__(self, input_shape, num_hidden_nodes, dropout):
        super().__init__()
        self.dropout = dropout
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_shape, num_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(num_hidden_nodes, 2)
        )
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

# Helper function for training and evaluating the classifier with ROC-AUC and accuracy metrics
def gen_classifier_roc(**cls_roc_args):
    inputs = cls_roc_args.get('inputs')
    correct = cls_roc_args.get('correct')
    artifact_name = cls_roc_args.get('artifact_name')
    device = cls_roc_args.get('device')
    cls_results_file_path = cls_roc_args.get('cls_results_file_path')
    cls_results_dir = cls_roc_args.get('cls_results_dir')

    # Generate original indices
    indices = list(range(len(inputs)))

    # Perform train-test split with shuffle=True (default)
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(inputs, correct.astype(int), indices, test_size=0.2, random_state=cls_roc_args.get('seed_val'))
    
    classifier_model = FFHallucinationClassifier(
        input_shape=X_train.shape[1], 
        num_hidden_nodes=cls_roc_args.get('num_hidden_nodes'), dropout=cls_roc_args.get('dropout_mlp')
    ).to(device)

    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=cls_roc_args.get('learning_rate'), weight_decay=cls_roc_args.get('weight_decay'))

    # Training loop
    for _ in range(1001):
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:cls_roc_args.get('batch_size')]
        pred = classifier_model(X_train[sample])
        loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()

    # Evaluation
    classifier_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(classifier_model(X_test), dim=1)
        prediction_classes = (pred[:, 1] > 0.5).type(torch.long).cpu()
        roc_auc = roc_auc_score(y_test.cpu(), pred[:, 1].cpu())
        accuracy = (prediction_classes.numpy() == y_test.cpu().numpy()).mean()

        # # Prepare data for saving
        # pred_list = pred[:, 1].cpu().numpy().tolist()  # Convert predictions to list

        # # Combine test indices from the original dataset and predictions
        # results = [{"index": idx, "prediction": p} for idx, p in zip(test_indices, pred_list)]

        pred_np = pred.cpu().numpy()
        entropy_list = []
        results = []

        for idx, prob_vec in zip(test_indices, pred_np):
            ent = entropy(prob_vec)
            entropy_list.append(ent)
            results.append({
                "index": idx,
                "prob_hallucination": float(prob_vec[1]),
                "prob_non_hallucination": float(prob_vec[0]),
                "entropy": float(ent)
            })

        avg_entropy = np.mean(entropy_list)

        # Save results to a JSON file
        file_name = f"classifier_results_{artifact_name}.json"
        with open(os.path.join(cls_results_dir, file_name), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        logging.info(f"Artifact: {artifact_name}")
        logging.info(f"ROC AUC: {roc_auc}, Accuracy: {accuracy}")
        logging.info(f"Results saved to {file_name}")

        with open(cls_results_file_path, "a") as f:
            f.write(f"Artifact: {artifact_name}\n")
            f.write(f"ROC AUC: {roc_auc}, Accuracy: {accuracy}\n\n")
            # f.write(f"Results saved to {file_name}\n")
        
        with open(cls_results_file_path, "a") as f:
            f.write(f"Classifier artifact: {artifact_name}\n")
            f.write(f"AVG Entropy: {avg_entropy}\n\n")

    return roc_auc, accuracy


# Individual functions for processing different data types
def process_logits(logits, correct, cls_results_file_path, cls_results_dir, device, args):
    first_logits = np.stack([sp.special.softmax(i) for i in logits])
    roc_auc, accuracy = gen_classifier_roc(
        inputs=first_logits,
        correct=correct,
        artifact_name="first_logits",
        device=device, 
        cls_results_file_path=cls_results_file_path,
        cls_results_dir=cls_results_dir,
        num_hidden_nodes=args.num_hidden_nodes,
        dropout_mlp=args.dropout_mlp,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed_val=args.seed_val
    )

    return roc_auc, accuracy


def process_first_fully_connected(first_fully_connected, correct, cls_results_file_path, cls_results_dir, device, args):
    results = {}
    for layer in range(first_fully_connected[0].shape[0]):
        layer_data = np.stack([i[layer] for i in first_fully_connected])
        layer_roc, layer_acc = gen_classifier_roc(
            inputs=layer_data,
            correct=correct,
            artifact_name=f"first_fully_connected_{layer}",
            device=device,
            cls_results_file_path=cls_results_file_path,
            cls_results_dir=cls_results_dir,
            num_hidden_nodes=args.num_hidden_nodes,
            dropout_mlp=args.dropout_mlp,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed_val=args.seed_val
        )
        results[f'first_fully_connected_roc_{layer}'] = layer_roc
        results[f'first_fully_connected_acc_{layer}'] = layer_acc

    return results

def process_token_pos_fully_connected_avg(token_pos_fully_connected_avg, correct, cls_results_file_path, cls_results_dir, device, args):
    results = {}
    for layer in range(token_pos_fully_connected_avg[0].shape[0]):
        layer_data = np.stack([i[layer] for i in token_pos_fully_connected_avg])
        layer_roc, layer_acc = gen_classifier_roc(
            inputs=layer_data,
            correct=correct,
            artifact_name=f"token_pos_fully_connected_avg_{layer}",
            device=device,
            cls_results_file_path=cls_results_file_path,
            cls_results_dir=cls_results_dir,
            num_hidden_nodes=args.num_hidden_nodes,
            dropout_mlp=args.dropout_mlp,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed_val=args.seed_val
        )
        results[f'token_pos_fully_connected_avg_roc_{layer}'] = layer_roc
        results[f'token_pos_fully_connected_avg_acc_{layer}'] = layer_acc
    return results


def process_first_attention(first_attention, correct, cls_results_file_path, cls_results_dir, device, args):
    results = {}
    for layer in range(first_attention[0].shape[0]):
        layer_data = np.stack([i[layer] for i in first_attention])
        layer_roc, layer_acc = gen_classifier_roc(
            inputs=layer_data,
            correct=correct,
            artifact_name=f"first_attention_{layer}",
            device=device,
            cls_results_file_path=cls_results_file_path,
            cls_results_dir=cls_results_dir,
            num_hidden_nodes=args.num_hidden_nodes,
            dropout_mlp=args.dropout_mlp,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed_val=args.seed_val
        )
        results[f'first_attention_roc_{layer}'] = layer_roc
        results[f'first_attention_acc_{layer}'] = layer_acc
    return results


def process_token_pos_attention_avg(token_pos_attention_avg, correct, cls_results_file_path, cls_results_dir, device, args):
    results = {}
    for layer in range(token_pos_attention_avg[0].shape[0]):
        layer_data = np.stack([i[layer] for i in token_pos_attention_avg])
        layer_roc, layer_acc = gen_classifier_roc(
            inputs=layer_data,
            correct=correct,
            artifact_name=f"token_pos_attention_avg_{layer}",
            device=device,
            cls_results_file_path=cls_results_file_path,
            cls_results_dir=cls_results_dir,
            num_hidden_nodes=args.num_hidden_nodes,
            dropout_mlp=args.dropout_mlp,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed_val=args.seed_val
        )
        results[f'token_pos_attention_avg_roc_{layer}'] = layer_roc
        results[f'token_pos_attention_avg_acc_{layer}'] = layer_acc
    return results

def compute_first_logits_metrics(all_first_logits):
    p1_p2_list = []
    entropy_list = []

    for logits in all_first_logits:
        probs = softmax(logits)
        
        top2 = np.partition(probs, -2)[-2:]
        p_sorted = np.sort(top2)[::-1]
        p1_p2 = p_sorted[0] - p_sorted[1]

        ent = entropy(probs)

        p1_p2_list.append(p1_p2)
        entropy_list.append(ent)
    
    avg_p1_p2 = np.mean(p1_p2_list)
    avg_entropy = np.mean(entropy_list)

    return avg_p1_p2, avg_entropy, p1_p2_list, entropy_list

def processing_dataset(data_category, device, args):
    results_dir = f"{args.base_results_dir}/ModelArti_{args.dataset_name}_results_v1/{args.model_name}_ModelArti_{data_category}_{args.language}_results"
    results_pickle_dir = os.path.join(results_dir, "pickle_files")

    cls_results_dir = os.path.join(results_dir, "classifier_results_latest")
    os.makedirs(cls_results_dir, exist_ok=True)
    cls_results_file_path = os.path.join(cls_results_dir, f"classifier_results.txt")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler(f"{cls_results_dir}/{args.model_name}_{data_category}_{args.language}_cls_results_log.txt"),
                    logging.StreamHandler(sys.stdout)
                ])
    
    logging.info(f"Using model: {args.model_name}, language: {args.language}, dataset: {args.dataset_name}, category: {data_category}")

    inference_results_pkl_files = list(Path(results_pickle_dir).rglob("*.pickle"))
    logging.info(f"Found {len(inference_results_pkl_files)} inference result files in {results_pickle_dir}.")

    lang_codes = {"english": "en", "hindi": "hi", "bengali": "bn", "deutsch": "de", "urdu": "ur"}

    all_results = {}
    time_now = time.time()

    if args.dataset_name == "mTREx":
        try: 
            correct = load_all_chunks(
                data_key='correct', 
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category, 
                max_num_samples=args.max_num_samples
            )

            first_logits = load_all_chunks(
                data_key='first_logits',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                max_num_samples=args.max_num_samples
            )

            first_fully_connected = load_all_chunks(
                data_key='first_fully_connected',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                max_num_samples=args.max_num_samples
            )

            token_pos_fully_connected_avg = load_all_chunks(
                data_key='token_pos_fully_connected_avg',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                max_num_samples=args.max_num_samples
            )

            first_attention = load_all_chunks(
                data_key='first_attention',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                max_num_samples=args.max_num_samples
            )

            token_pos_attention_avg = load_all_chunks(
                data_key='token_pos_attention_avg',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                max_num_samples=args.max_num_samples
            )

        except Exception as e:
            logging.error(f"Error processing data for {data_category} category in {args.dataset_name} dataset: {e}")
    
    elif args.dataset_name == "GMMLU":
        Global_MMLU = load_dataset("CohereForAI/Global-MMLU", lang_codes[args.language.lower()])
        Global_MMLU.set_format("pandas")
        data = Global_MMLU['test'][:]
        # Filter data by subject_category
        category_data = data[data["subject_category"] == data_category]
        # Get unique subjects within this category
        subjects = category_data["subject"].unique()
 
        try:
            correct = load_all_chunks(
                data_key='correct', 
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                gmmlu_subjects=subjects
            )

            first_logits = load_all_chunks(
                data_key='first_logits',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                gmmlu_subjects=subjects
            )

            first_fully_connected = load_all_chunks(
                data_key='first_fully_connected',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                gmmlu_subjects=subjects
            )

            token_pos_fully_connected_avg = load_all_chunks(
                data_key='token_pos_fully_connected_avg',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                gmmlu_subjects=subjects
            )

            first_attention = load_all_chunks(
                data_key='first_attention',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                gmmlu_subjects=subjects
            )

            token_pos_attention_avg = load_all_chunks(
                data_key='token_pos_attention_avg',
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name,
                data_category=data_category,
                gmmlu_subjects=subjects
            )

        except Exception as e:
            logging.error(f"Error processing data for {data_category} category in {args.dataset_name} dataset: {e}")
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}. Supported datasets are 'mTREx' and 'GMMLU'.")
    
    correct_count = correct.count(1)
    # correct = np.array(correct)

    hallu_true = [1-x for x in correct]
    correct = np.array(hallu_true)

    avg_p1_p2, avg_entropy, p1_p2_list, entropy_list = compute_first_logits_metrics(first_logits)

    classifier_results = {}
    # Process logits data and add results
    classifier_results['first_logits_roc'], classifier_results['first_logits_acc'] = process_logits(
        logits=first_logits, 
        correct=correct, 
        cls_results_file_path=cls_results_file_path, 
        cls_results_dir=cls_results_dir, 
        device=device,
        args=args
    )

    # Process fully connected layer data and add results
    classifier_results.update(process_first_fully_connected(
        first_fully_connected=first_fully_connected, 
        correct=correct, 
        cls_results_file_path=cls_results_file_path, 
        cls_results_dir=cls_results_dir, 
        device=device,
        args=args
    ))

    # Process attention layer data and add results
    classifier_results.update(process_first_attention(
        first_attention=first_attention, 
        correct=correct, 
        cls_results_file_path=cls_results_file_path, 
        cls_results_dir=cls_results_dir, 
        device=device,
        args=args
    ))

    # Process token position fully connected average layer data and add results
    classifier_results.update(process_token_pos_fully_connected_avg(
        token_pos_fully_connected_avg=token_pos_fully_connected_avg, 
        correct=correct, 
        cls_results_file_path=cls_results_file_path, 
        cls_results_dir=cls_results_dir, 
        device=device,
        args=args
    ))

    # Process token position attention average layer data and add results
    classifier_results.update(process_token_pos_attention_avg(
        token_pos_attention_avg=token_pos_attention_avg, 
        correct=correct, 
        cls_results_file_path=cls_results_file_path, 
        cls_results_dir=cls_results_dir, 
        device=device,
        args=args
    ))

    # Store the results
    all_results[f"ModelArti_{args.model_name}_{args.dataset_name}_{data_category}_{args.language}"] = classifier_results

    logging.info(f"Total correct answers in {data_category} category for {args.model_name} in {args.language}: {correct_count}")

    with open(cls_results_file_path, 'a') as f:
        f.write(f"Average P1-P2 difference from first generated token for model {args.model_name}\n")
        f.write(f"AVG (P1-P2): {avg_p1_p2}\n\n")
        f.write(f"Average Entropy from first generated token for model {args.model_name}\n")
        f.write(f"AVG (Entropy): {avg_entropy}\n\n")

    # Save the results to a text file
    with open(cls_results_file_path, 'a') as f:
        f.write(f"Total correct answers in {data_category} category for {args.model_name} in {args.language}: {correct_count}\n")
        f.write(f"Classifier results:\n")
    for artifact, metrics in all_results.items():
        logging.info(f"{artifact}: {metrics}")
        with open(cls_results_file_path, 'a') as f:
            f.write(f"{artifact}: {metrics}\n")
        
    logging.info(f"Average P1-P2 difference from first generated token: {avg_p1_p2}")
    logging.info(f"Average Entropy from first generated token: {avg_entropy}")

    logging.info(f"Time taken: {(time.time()-time_now)/3600} hours for {data_category} category of {args.dataset_name} dataset with {args.model_name} model in {args.language} language.")


def main(args):
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")

    if args.dataset_name == "mTREx":
        data_category_list = ["capitals", "country", "official_language"]
    elif args.dataset_name == "GMMLU":
        data_category_list = ["STEM", "Humanities"]
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}. Supported datasets are 'mTREx' and 'GMMLU'.")
    
    for data_category in data_category_list:
        processing_dataset(
            data_category=data_category, 
            device=device,
            args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluating classifier over model artifacts across different LLMs and languages for both G-MMLU and mTREx datasets.")
    parser.add_argument("--gpu_index", type=str, help="Give one aviable GPU index to utilize for inference", required=False, default="0")
    parser.add_argument("--model_name", type=str, choices=["Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct", "Mistral-7B-Instruct-v0.3", "Mistral-Small-24B-Instruct-2501"], help="Name of the model to use for inference", required=True)
    parser.add_argument("--language", type=str, choices=["English", "Hindi", "Bengali", "Deutsch", "Urdu"], help="Language to use for inference", required=True)
    parser.add_argument("--max_num_samples", type=int, help="Maximum number of samples to process from the dataset", required=False, default=2500)
    parser.add_argument("--num_hidden_nodes", type=int, help="Number of hidden nodes in the MLP layers", required=False, default=256)
    parser.add_argument("--batch_size", type=int, help="Batch size for inference", required=False, default=128)
    parser.add_argument("--dropout_mlp", type=float, help="Dropout rate for MLP layers", required=False, default=0.5)
    parser.add_argument("--dropout_gru", type=float, help="Dropout rate for GRU layers", required=False, default=0.25)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer", required=False, default=1e-4)
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the optimizer", required=False, default=1e-2)
    parser.add_argument("--base_results_dir", type=str, help="Base directory to save results", required=False, default="/data/debtanu/Research_work/Results_ModelArti_Rebuttal_Entropy")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use for inference", choices=["mTREx", "GMMLU"], required=True)
    parser.add_argument("--seed_val", type=int, help="Seed value for reproducibility", required=False, default=42)
    args = parser.parse_args()
    main(args)
