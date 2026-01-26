"""## Import necessary libraries and modules"""

import pickle, numpy as np, scipy as sp, torch, random, os, json, sys, logging, argparse, time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datasets import load_dataset

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


# =================== New Multi-Lingual Training/Eval ===================

def train_classifier(X_train, y_train, input_shape, device, args):
    classifier_model = FFHallucinationClassifier(
        input_shape=input_shape,
        num_hidden_nodes=args.num_hidden_nodes,
        dropout=args.dropout_mlp
    ).to(device)

    X_train = torch.tensor(X_train).to(device)
    y_train = torch.tensor(y_train).to(torch.long).to(device)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for _ in range(1001):
        optimizer.zero_grad()
        sample = torch.randperm(X_train.shape[0])[:args.batch_size]
        pred = classifier_model(X_train[sample])
        loss = torch.nn.functional.cross_entropy(pred, y_train[sample])
        loss.backward()
        optimizer.step()

    return classifier_model

def eval_classifier(cls_model, X_test, y_test, artifact_name, cls_results_dir, cls_results_file_path, device):
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(torch.long).to(device)

    test_indices = list(range(len(y_test)))

    cls_model.eval()
    with torch.no_grad():
        pred = torch.nn.functional.softmax(cls_model(X_test), dim=1)
        prediction_classes = (pred[:, 1] > 0.5).type(torch.long).cpu()
        roc_auc = roc_auc_score(y_test.cpu(), pred[:, 1].cpu())
        accuracy = (prediction_classes.numpy() == y_test.cpu().numpy()).mean()

        # Prepare data for saving
        pred_list = pred[:, 1].cpu().numpy().tolist()  # Convert predictions to list

        # Combine test indices from the original dataset and predictions
        results = [{"index": idx, "prediction": p} for idx, p in zip(test_indices, pred_list)]

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
            f.write(f"Results saved to {file_name}\n")
    return roc_auc, accuracy

# =================== Data Preparation Helpers ===================

def prepare_labels(correct):
    hallu_true = [1-x for x in correct]
    return np.array(hallu_true)

# =================== Main Processing ===================

def process_artifacts(language, data_category, args):
    cls_results_dir = f"{args.base_results_dir}/ModelArti_{args.dataset_name}_results_v1/{args.model_name}_multilingual_all_lang_train/{data_category}_{language}"
    os.makedirs(cls_results_dir, exist_ok=True)
    cls_results_file_path = os.path.join(cls_results_dir, "classifier_performance.txt")

    results_pickle_dir = f"{args.base_results_dir}/ModelArti_{args.dataset_name}_results_v1/{args.model_name}_ModelArti_{data_category}_{language}_results/pickle_files"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler(f"{cls_results_dir}/{args.model_name}_{data_category}_{language}_multilingual_cls_results_log.txt"),
                    logging.StreamHandler(sys.stdout)
                ])

    logging.info(f"Processing model: {args.model_name}, dataset: {args.dataset_name}, language: {language}, data_category: {data_category}")

    inference_results_pkl_files = list(Path(results_pickle_dir).rglob("*.pickle"))
    logging.info(f"Found {len(inference_results_pkl_files)} inference result files in {results_pickle_dir}.")

    lang_codes = {"english": "en", "hindi": "hi", "bengali": "bn", "deutsch": "de", "urdu": "ur"}

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
            first_attention = load_all_chunks(
                data_key='first_attention', 
                chunk_dir=results_pickle_dir,
                dataset_name=args.dataset_name, 
                data_category=data_category,
                max_num_samples=args.max_num_samples
            )
        except Exception as e:
            logging.error(f"Error processing data for {data_category} category in {args.dataset_name} dataset: {e}")

    elif args.dataset_name == "GMMLU":
        Global_MMLU = load_dataset("CohereForAI/Global-MMLU", lang_codes[language.lower()])
        Global_MMLU.set_format("pandas")
        data = Global_MMLU['test'][:]
        category_data = data[data["subject_category"] == data_category]
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
            first_attention = load_all_chunks(
                data_key='first_attention', 
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
    correct = prepare_labels(correct)
    first_logits = np.stack([sp.special.softmax(i) for i in first_logits])

    return correct, first_logits, first_fully_connected, first_attention, cls_results_dir, cls_results_file_path

# ---------------- Main ----------------

def main(args):
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")

    if args.dataset_name == "mTREx":
        data_category_list = ["capitals", "country", "official_language"]
        languages = ["English", "Hindi", "Bengali", "Deutsch", "Urdu"]
    elif args.dataset_name == "GMMLU":
        data_category_list = ["STEM", "Humanities"]
        languages = ["English", "Hindi", "Bengali", "Deutsch"]
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}.")

    for data_category in data_category_list:
        all_train_logits, all_train_labels = [], []
        all_fc_train, all_attn_train = None, None
        all_test_sets = {}

        # Collect train/test splits for each language
        for lang in languages:
            correct, logits, fc, attn, cls_dir, cls_file = process_artifacts(lang, data_category, args)
            X_tr_logits, X_te_logits, y_tr, y_te = train_test_split(logits, correct, test_size=args.test_size, random_state=args.seed_val)

            # FC + ATT
            fc_train, fc_test, attn_train, attn_test = [], [], [], []
            for layer in range(fc[0].shape[0]):
                layer_data = np.stack([i[layer] for i in fc])
                X_tr, X_te, _, _ = train_test_split(layer_data, correct, test_size=args.test_size, random_state=args.seed_val)
                fc_train.append(X_tr)
                fc_test.append(X_te)
            for layer in range(attn[0].shape[0]):
                layer_data = np.stack([i[layer] for i in attn])
                X_tr, X_te, _, _ = train_test_split(layer_data, correct, test_size=args.test_size, random_state=args.seed_val)
                attn_train.append(X_tr)
                attn_test.append(X_te)

            # accumulate training data
            all_train_logits.append(X_tr_logits)
            all_train_labels.append(y_tr)
            if all_fc_train is None:
                all_fc_train = [[] for _ in range(len(fc_train))]
                all_attn_train = [[] for _ in range(len(attn_train))]
            for l in range(len(fc_train)):
                all_fc_train[l].append(fc_train[l])
            for l in range(len(attn_train)):
                all_attn_train[l].append(attn_train[l])

            # store test splits per language
            all_test_sets[lang] = (X_te_logits, y_te, fc_test, attn_test, cls_dir, cls_file)

        # merge train sets across all languages
        X_train_logits = np.concatenate(all_train_logits)
        y_train = np.concatenate(all_train_labels)
        fc_train_merged = [np.concatenate(layers) for layers in all_fc_train]
        attn_train_merged = [np.concatenate(layers) for layers in all_attn_train]

        # --- shuffle after merging ---
        rng = np.random.default_rng(args.seed_val)
        indices = rng.permutation(len(y_train))

        X_train_logits = X_train_logits[indices]
        y_train = y_train[indices]
        fc_train_merged = [layer[indices] for layer in fc_train_merged]
        attn_train_merged = [layer[indices] for layer in attn_train_merged]

        # Train models
        models = {}
        models['logits'] = train_classifier(X_train_logits, y_train, X_train_logits.shape[1], device, args)
        models['fc'] = [train_classifier(fc_train_merged[l], y_train, fc_train_merged[l].shape[1], device, args)
                        for l in range(len(fc_train_merged))]
        models['attn'] = [train_classifier(attn_train_merged[l], y_train, attn_train_merged[l].shape[1], device, args)
                          for l in range(len(attn_train_merged))]
        
        # Evaluate on 20% split of each language
        for lang, (X_te_logits, y_te, fc_test, attn_test, cls_dir, cls_file) in all_test_sets.items():
            eval_classifier(models['logits'], X_te_logits, y_te, f"logits_{lang}_{data_category}", cls_dir, cls_file, device)
            for l, X_te in enumerate(attn_test):
                eval_classifier(models['attn'][l], X_te, y_te, f"attn_{l}_{lang}_{data_category}", cls_dir, cls_file, device)
            for l, X_te in enumerate(fc_test):
                eval_classifier(models['fc'][l], X_te, y_te, f"fc_{l}_{lang}_{data_category}", cls_dir, cls_file, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual hallucination classifier: Train on 80% of all languages, test on 20% of each.")
    parser.add_argument("--gpu_index", type=str, help="Give one aviable GPU index to utilize for inference", required=False, default="0")
    parser.add_argument("--model_name", type=str, choices=["Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct", "Mistral-7B-Instruct-v0.3", "Mistral-Small-24B-Instruct-2501"], help="Name of the model to use for inference", required=True)
    parser.add_argument("--max_num_samples", type=int, help="Maximum number of samples to process from the dataset", required=False, default=2500)
    parser.add_argument("--num_hidden_nodes", type=int, help="Number of hidden nodes in the MLP layers", required=False, default=256)
    parser.add_argument("--batch_size", type=int, help="Batch size for inference", required=False, default=128)
    parser.add_argument("--dropout_mlp", type=float, help="Dropout rate for MLP layers", required=False, default=0.5)
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer", required=False, default=1e-4)
    parser.add_argument("--weight_decay", type=float, help="Weight decay for the optimizer", required=False, default=1e-2)
    parser.add_argument("--base_results_dir", type=str, help="Base directory to save results", required=False, default="/data/debtanu/Research_work/Results_Part_1_ModelArti")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use for inference", choices=["mTREx", "GMMLU"], required=True)
    parser.add_argument("--test_size", type=float, help="Proportion of the dataset to include in the test split", required=False, default=0.2)
    parser.add_argument("--seed_val", type=int, help="Seed value for reproducibility", required=False, default=42)
    args = parser.parse_args()
    main(args)
    