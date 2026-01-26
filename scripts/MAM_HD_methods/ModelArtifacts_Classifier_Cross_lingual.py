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
    

# =================== New Cross-Lingual Training/Eval ===================

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

def process_artifacts(language, data_category, device, args, models=None):
    cls_results_dir = f"{args.base_results_dir}/ModelArti_{args.dataset_name}_results_v1/{args.model_name}_crosslingual/{data_category}_{language}"
    os.makedirs(cls_results_dir, exist_ok=True)

    cls_results_file_path = os.path.join(cls_results_dir, "classifier_performance.txt")

    results_pickle_dir = f"{args.base_results_dir}/ModelArti_{args.dataset_name}_results_v1/{args.model_name}_ModelArti_{data_category}_{language}_results/pickle_files"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler(f"{cls_results_dir}/{args.model_name}_{data_category}_{language}_crosslingual_cls_results_log.txt"),
                    logging.StreamHandler(sys.stdout)
                ])

    logging.info(f"Processing model: {args.model_name}, dataset: {args.dataset_name}, language: {language}, data_category: {data_category}")

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

    # 80-20 split for reproducibility
    X_train_logits, X_test_logits, y_train, y_test = train_test_split(first_logits, correct, test_size=args.test_size, random_state=args.seed_val)
    fc_train, fc_test = [], []
    for layer in range(first_fully_connected[0].shape[0]):
        layer_data = np.stack([i[layer] for i in first_fully_connected])
        X_tr, X_te, _, _ = train_test_split(layer_data, correct, test_size=args.test_size, random_state=args.seed_val)
        fc_train.append(X_tr)
        fc_test.append(X_te)
    attn_train, attn_test = [], []
    for layer in range(first_attention[0].shape[0]):
        layer_data = np.stack([i[layer] for i in first_attention])
        X_tr, X_te, _, _ = train_test_split(layer_data, correct, test_size=args.test_size, random_state=args.seed_val)
        attn_train.append(X_tr)
        attn_test.append(X_te)
    
    classifier_results = {}

    if models is None:
        # Training stage (English only)
        models = {}
        models['logits'] = train_classifier(X_train_logits, y_train, X_train_logits.shape[1], device, args)
        models['fc'] = []
        for layer, X_tr in enumerate(fc_train):
            models['fc'].append(train_classifier(X_tr, y_train, X_tr.shape[1], device, args))
        models['attn'] = []
        for layer, X_tr in enumerate(attn_train):
            models['attn'].append(train_classifier(X_tr, y_train, X_tr.shape[1], device, args))
        # Evaluate on English 20%
        classifier_results['logits'] = eval_classifier(models['logits'], X_test_logits, y_test, f"logits_{language}_{data_category}", cls_results_dir, cls_results_file_path, device)
        for layer, X_te in enumerate(attn_test):
            classifier_results[f'attn_layer{layer}'] = eval_classifier(models['attn'][layer], X_te, y_test, f"attn_{layer}_{language}_{data_category}", cls_results_dir, cls_results_file_path, device)
        for layer, X_te in enumerate(fc_test):
            classifier_results[f'fc_layer{layer}'] = eval_classifier(models['fc'][layer], X_te, y_test, f"fc_{layer}_{language}_{data_category}", cls_results_dir, cls_results_file_path, device)
        return models
    else:
        # Evaluation stage (non-English)
        classifier_results['logits'] = eval_classifier(models['logits'], X_test_logits, y_test, f"logits_{language}_{data_category}", cls_results_dir, cls_results_file_path, device)
        for layer, X_te in enumerate(attn_test):
            classifier_results[f'attn_layer{layer}'] = eval_classifier(models['attn'][layer], X_te, y_test, f"attn_{layer}_{language}_{data_category}", cls_results_dir, cls_results_file_path, device)
        for layer, X_te in enumerate(fc_test):
            classifier_results[f'fc_layer{layer}'] = eval_classifier(models['fc'][layer], X_te, y_test, f"fc_{layer}_{language}_{data_category}", cls_results_dir, cls_results_file_path, device)
        all_results[f"ModelArti_{args.model_name}_{args.dataset_name}_{data_category}_{language}"] = classifier_results
        logging.info(f"Total correct answers in {data_category} category for {args.model_name} in {language}: {correct_count} out of {len(correct)}")

        with open(cls_results_file_path, "a") as f:
            f.write(f"Total correct answers in {data_category} category for {args.model_name} in {language}: {correct_count} out of {len(correct)}\n")
            f.write(f"Classifier results:\n")
            for artifact, (roc_auc, accuracy) in all_results[f"ModelArti_{args.model_name}_{args.dataset_name}_{data_category}_{language}"].items():
                f.write(f"{artifact} - ROC AUC: {roc_auc}, Accuracy: {accuracy}\n")

        logging.info(f"Completed processing for model: {args.model_name}, dataset: {args.dataset_name}, language: {language}, data_category: {data_category} in {(time.time()-time_now)/3600} hours.")

def main(args):
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")

    if args.dataset_name == "mTREx":
        data_category_list = ["capitals", "country", "official_language"]
        target_languages = ["Bengali", "Hindi", "Deutsch", "Urdu"]
    elif args.dataset_name == "GMMLU":
        data_category_list = ["STEM", "Humanities"]
        target_languages = ["Bengali", "Hindi", "Deutsch"]
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}.")        

    for data_category in data_category_list:
        # 1. Train on English (80%) + Evaluate on English (20%)
        cls_models = process_artifacts("English", data_category, device, args, models=None)

        # 2. Evaluate on others (20%)
        for lang in target_languages:
            process_artifacts(lang, data_category, device, args, models=cls_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-lingual hallucination classifier: train on 80% English, test on 20% of English and others.")
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
