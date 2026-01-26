"""## Import necessary libraries and modules"""

import os, collections, json, sys, gc, re, argparse, time, logging, torch, numpy as np, pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

"""## Define functions for processing"""

def format_chat_prompt(**chat_args):
    dataset_name = chat_args.get('dataset_name')
    if dataset_name == "mTREx":
        messages = [
            {"role": "system", "content": chat_args.get('system_message')},
            {"role": "user", "content": chat_args.get('user_message')},
            {"role": "assistant", "content": chat_args.get('asst_message')}
        ]
    elif dataset_name == "GMMLU":
        messages = [{"role": "system", "content": chat_args.get('system_message')}]
        for ex in chat_args.get('examples'):
            messages.append({"role": "user", "content": ex["user"]+"\n"})
            messages.append({"role": "assistant", "content": ex["assistant"]+"\n"})
        messages.append({"role": "user", "content": chat_args.get('actual_question')})
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return messages


def get_tokenized_prompt(**token_prompt_args):
    tokenizer = token_prompt_args.get('tokenizer')
    messages = format_chat_prompt(**token_prompt_args)
    chat_template_message = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True, 
    )
    return chat_template_message


def check_similarity(text1, text2, sbert_model):
    inputs = sbert_model.encode([text1, text2], convert_to_tensor=True)
    cosine_scores = torch.nn.functional.cosine_similarity(inputs[0], inputs[1], dim=0)
    return cosine_scores.item()


def get_semantic_ids(strings_list, cutoff_score, sbert_model):
    def are_equivalent(text1, text2):
        value = check_similarity(text1, text2, sbert_model)
        if value > cutoff_score:
            semantically_equivalent = True
        else:
            semantically_equivalent = False
        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum':
            logsumexp_value = np.log(np.sum(np.exp(id_log_likelihoods))) - 5.0
        elif agg == 'sum_normalized':
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        elif agg == 'mean':
            logsumexp_value = np.log(np.mean(np.exp(id_log_likelihoods)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)` where i are the is the sequence
    likelihood, i.e. the average token likelihood.
    """
    entropy = -np.sum(log_probs) / len(log_probs)
    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


def generate_responses(**gen_res_args):
    num_generations = gen_res_args.get('num_generations')
    max_new_tokens = gen_res_args.get('max_new_tokens')
    tokenizer = gen_res_args.get('tokenizer')
    model = gen_res_args.get('model')
    device = gen_res_args.get('device')

    tokenized_inputs = tokenizer(text = gen_res_args.get('question'), return_tensors="pt", padding = True).to(device)
    start_pos = tokenized_inputs["input_ids"].size(dim=-1)

    # We sample base response (zero temperature) for estimating model accuracy
    with torch.no_grad():
        model_outputs_1 = model.generate(input_ids=tokenized_inputs["input_ids"], attention_mask=tokenized_inputs["attention_mask"],
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True, 
            output_scores=True, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id
        )

    # Remove input prompt from generated output
    low_temp_response = tokenizer.decode(model_outputs_1.sequences[0][start_pos:], skip_special_tokens=True)
    low_temp_response = low_temp_response.replace("\n", " ").strip()
    
    full_responses = []
    
    with torch.no_grad():
        model_outputs = model.generate(input_ids=tokenized_inputs["input_ids"], attention_mask=tokenized_inputs["attention_mask"],
            do_sample=True, 
            temperature=1, 
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, 
            output_scores=True, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id, 
            num_return_sequences=num_generations
        )
    
    # Compute transition scores for all generated sequences
    transition_scores = model.compute_transition_scores(model_outputs.sequences, model_outputs.scores, normalize_logits=True)

    for i in range(num_generations):
        # Extract generated tokens after the input prompt
        generated_tokens_ids = model_outputs.sequences[i][start_pos:]
        
        # Find the index of the first EOS token
        eos_indices = (generated_tokens_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            first_eos_idx = eos_indices[0].item()
            # Include tokens up to and including the first EOS
            filtered_tokens_ids = generated_tokens_ids[:first_eos_idx + 1]
            filtered_log_likelihoods = [transition_scores[i][j].item() for j in range(first_eos_idx + 1)]
        else:
            # No EOS token found (unlikely with eos_token_id set), include all tokens
            filtered_tokens_ids = generated_tokens_ids
            filtered_log_likelihoods = [transition_scores[i][j].item() for j in range(len(transition_scores[i]))]

        # Decode the response and store with log likelihoods
        response = tokenizer.decode(filtered_tokens_ids, skip_special_tokens=True)
        response = response.replace("\n", " ").strip()

        full_responses.append((response, filtered_log_likelihoods))

        del generated_tokens_ids
        gc.collect()
        
    del model_outputs
    gc.collect()
    
    return (low_temp_response, full_responses)


def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def post_process_mtrex_response(target, str_response, user_message):
    # Clean the target by removing unwanted characters
    cleaned_target = re.sub(r'[\'\"\(\)]', '', target)
    # Split the target into possible answers using '+' as the delimiter
    target_phrases = [phrase.strip().lower() for phrase in cleaned_target.split("<OR>")]
    # Check if the model's response matches any of the target phrases
    correct = any(phrase in str_response.lower() for phrase in target_phrases)

    logging.info(f"question: {user_message.strip()}")
    logging.info(f"target phrases: {target_phrases}")
    logging.info(f"str response: {str_response}")
    logging.info(f"correct: {correct}\n")

    return target_phrases, correct

def post_process_GMMLU_response(correct_choice, str_response):
        decoded_tokens = str_response.split()
        base_options = ["A", "B", "C", "D"]
        options = base_options.copy()
        for op in base_options:
            options += [f"{op}.", f"{op}:", f"{op},", f"{op};"]

        # logging.info(f"options: {options}")
        
        correct = 0
        if decoded_tokens and decoded_tokens[0] not in options: correct = 0
        elif decoded_tokens and decoded_tokens[0][0] == correct_choice: correct = 1
        logging.info(f"decoded_tokens: {decoded_tokens}")
        logging.info(f"correct_choice: {correct_choice}")
        logging.info(f"correct: {correct}\n")
        
        return decoded_tokens, correct


def answer_question(**ans_ques_args):
    tokenizer = ans_ques_args.get('tokenizer')
    sbert_model = ans_ques_args.get('sbert_model')
    model = ans_ques_args.get('model')
    device = ans_ques_args.get('device')
    generations = ans_ques_args.get('generations')
    lang = ans_ques_args.get('lang')
    dataset_name = ans_ques_args.get('dataset_name')

    if dataset_name == "mTREx":
        prompts = load_prompts(os.path.join(ans_ques_args.get('prompts_dir'), f"{dataset_name}_prompts.json"))
        lang_prompts = prompts[lang]
        system_message = lang_prompts['system']
        dataset_category_prompts = lang_prompts[ans_ques_args.get('mTREx_category')]
        source = ans_ques_args.get('source')
        target = ans_ques_args.get('target')
        user_message = dataset_category_prompts['user'].format(source=source)
        question = user_message
        asst_message = dataset_category_prompts['asst']
        chat_str = get_tokenized_prompt(
            system_message=system_message, 
            user_message=user_message, 
            asst_message=asst_message, 
            tokenizer=tokenizer,
            dataset_name=dataset_name)
        
    elif dataset_name == "GMMLU":
        choices  = ans_ques_args.get('choices')
        question = ans_ques_args.get('question')
        if len(choices) < 4: return 0
        formatted_question = (
            f"Q: {question}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
        )
        prompts = load_prompts(os.path.join(ans_ques_args.get('prompts_dir'), f"{dataset_name}_prompts.json"))
        lang_prompts = prompts[lang]
        system_message = lang_prompts['system'].format(sub=ans_ques_args.get('GMMLU_category'))
        chat_str = get_tokenized_prompt(system_message=system_message, 
                                        actual_question = formatted_question,
                                        examples = ans_ques_args.get('examples'),
                                        tokenizer=tokenizer,
                                        dataset_name=dataset_name)

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")


    str_response, full_responses = generate_responses(
                                    question=chat_str, 
                                    num_generations=ans_ques_args.get('num_generations'), 
                                    max_new_tokens=ans_ques_args.get('max_new_tokens'), 
                                    tokenizer=tokenizer,
                                    model=model,
                                    device=device)

    str_responses = [r[0] for r in full_responses]
    log_liks = [r[1] for r in full_responses]

    if dataset_name == "mTREx":
        target_phrases, correct = post_process_mtrex_response(target=target, str_response=str_response, user_message=user_message)
        final_target = target_phrases
    elif dataset_name == "GMMLU":
        final_target = ans_ques_args.get('correct_choice')
        decoded_tokens, correct = post_process_GMMLU_response(correct_choice=final_target, str_response=str_response)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    entropies = defaultdict(list)
    semantic_ids = get_semantic_ids(str_responses, ans_ques_args.get('cutoff_score'), sbert_model)
    entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))
    for agg_name, agg_func in zip(['', '_sum'], [np.mean, np.sum]):
        log_liks_agg = [agg_func(log_lik) for log_lik in log_liks]

        # Compute standard entropy
        entropies['regular_entropy' + agg_name].append(predictive_entropy(log_liks_agg))

        # Compute semantic entropies with summing and with averaging probabilities within the cluster
        cluster_agg_names = ['', '_sum-normalized', '_sum-normalized-rao', '_cmean']
        cluster_aggs = ['sum', 'sum_normalized', 'sum_normalized', 'mean']
        for cluster_agg_name, cluster_agg in zip(cluster_agg_names, cluster_aggs):
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg=cluster_agg)
            name = 'semantic_entropy' + agg_name + cluster_agg_name

            if cluster_agg_name != '_sum-normalized-rao':
                pe = predictive_entropy(log_likelihood_per_semantic_id)
            else:
                pe = predictive_entropy_rao(log_likelihood_per_semantic_id)

            entropies[name].append(pe)

    result = {
        'question': question,
        'target': final_target,
        'response': str_response,
        'correct': correct,
        'sampled_responses': str_responses,
        'log_likelihoods': log_liks,
        'semantic_ids': semantic_ids,
        'entropies': entropies,
    }

    generations.append(result)

    return correct, entropies, generations

def load_model_and_tokenizer(model_name, hf_cache_dir, max_memory):
    model_org = "meta-llama"
    if model_name in ("Mistral-7B-Instruct-v0.3", "Mistral-Small-24B-Instruct-2501"):
        model_org = "mistralai"
    model_repo = f"{model_org}/{args.model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_repo, padding_side = "left") # LLMs aren't trained to continue generation from padding tokens, which means the padding_side() parameter needs to be set to the left of the input.
    tokenizer.pad_token_id = tokenizer.eos_token_id # Most Tokenizer does not have PAD token inherently
    model = AutoModelForCausalLM.from_pretrained(model_repo, cache_dir=hf_cache_dir, torch_dtype=torch.bfloat16, 
                                            trust_remote_code=True,
                                            device_map="auto",
                                            max_memory=max_memory)
    model.eval()
    return model, tokenizer

def load_sentence_transformer(model_name, device):
    sbert_model = SentenceTransformer(model_name, device=device)
    sbert_model.eval()
    return sbert_model

def set_max_memory(gpu_index):
    max_mem = {str(i): "0GiB" for i in range(8)}
    if gpu_index in ["0", "1", "2", "3", "4", "5", "6", "7"]:
        max_mem[gpu_index] = "135GiB"  # Set a high limit for the specified GPU
    else:
        raise ValueError(f"Invalid GPU index: {gpu_index}. Must be one of '0' to '7'.")
    max_memory = {
        0: max_mem["0"],
        1: max_mem["1"],
        2: max_mem["2"],
        3: max_mem["3"],
        4: max_mem["4"],
        5: max_mem["5"],
        6: max_mem["6"],
        7: max_mem["7"],
        "cpu": "1GiB"  # Set a very low limit for CPU so that no layers are assigned there.
    }
    return max_memory

def load_dataset_mTREx(**kwargs):
    dataset_df = pd.read_csv(f"{kwargs.get('data_dir')}/{kwargs.get('lang')}/{kwargs.get('mTREx_category')}.csv")
    data = []
    for idx in range(kwargs.get('max_num_samples')):
        data.append((dataset_df.iloc[idx]['source'], dataset_df.iloc[idx]['target']))
        if idx >= len(dataset_df) - 1:
            break
    final_data = []
    for src, tgt in data:
        final_data.append((src, tgt[0:]))
    
    return final_data

def compute_auroc(uncertain_arr, scores_by_key, output_file_path):
    for key, values in scores_by_key.items():
        try:
            auroc = roc_auc_score(uncertain_arr, values)
            logging.info(f"AUROC for {key}: {auroc}")
            with open(output_file_path,'a', encoding='utf-8') as f:
                logging.info(f"AUROC for {key}: {auroc}")
                print(f"AUROC for {key}: {auroc}", file=f)
        except Exception as e:
            logging.info(f"Error computing AUROC for key {key}: {e}")


def saving_scores(uncertain_arr, output_file_path, generations, actual_questions, results_file_path):
    correct_count = uncertain_arr.count(0)
    logging.info(f"Number of true points: {correct_count}")
    logging.info(f"Total questions processed: {actual_questions}")

    with open(output_file_path,'a', encoding='utf-8') as f:
        logging.info(f"Number of true points: {correct_count}")
        logging.info(f"Total questions processed: {actual_questions}")
        print(f"Number of true points: {correct_count}", file=f)
        print(f"Total questions processed: {actual_questions}", file=f)

    with open(results_file_path, 'w', encoding='utf-8') as f:
            json.dump(generations, f, ensure_ascii=False, indent=4)

    logging.info(f"Results saved to {results_file_path}")


def processing_dataset(data_category, model, tokenizer, sbert_model, device, args):
    RESULTS_DIR = f"{args.base_results_dir}/SE_{args.dataset_name}_results_v1/{args.model_name}_SE_{data_category}_{args.language}_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file_path = os.path.join(RESULTS_DIR, f'{args.dataset_name}_{args.model_name}_{data_category}_{args.language}.json')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler(f"{RESULTS_DIR}/{args.model_name}_SE_{data_category}_{args.language}_results_log.txt"),
                    logging.StreamHandler(sys.stdout)
                ])
    logging.info(f"Using model: {args.model_name}, language: {args.language}, dataset: {args.dataset_name}, category: {data_category}")
    
    # We'll aggregate entropy scores by key (excluding the ones we don't need)
    excluded_keys = {"regular_entropy", "regular_entropy_sum"}

    output_file_path = os.path.join(RESULTS_DIR, f'{args.dataset_name}_{args.model_name}_{data_category}_{args.language}.txt')
    with open(output_file_path,'w', encoding='utf-8') as f:
        logging.info(f"Evaluating {data_category} category")
        print(f"Evaluating {data_category} category", file=f)

    actual_questions = 0
    uncertain_arr = []
    generations = []
    scores_by_key = collections.defaultdict(list)
    time_now = time.time()
    if args.dataset_name == "mTREx":
        data = load_dataset_mTREx(data_dir = args.mTREx_data_dir, lang = args.language, mTREx_category = data_category, 
                                    max_num_samples = args.max_num_samples,
                                    seed_val = args.seed_val)
        for idx in tqdm(range(len(data)), desc="Processing questions"):
            try:
                actual_questions += 1
                source = data[idx][0]
                target = data[idx][1]

                correct, entropies, generations = answer_question(
                    source=source, 
                    target=target, 
                    tokenizer=tokenizer, 
                    sbert_model=sbert_model,
                    model=model, 
                    device=device,
                    generations=generations, 
                    cutoff_score=args.cutoff_score, 
                    mTREx_category=data_category, 
                    lang=args.language,
                    prompts_dir=args.prompts_dir,
                    dataset_name=args.dataset_name,
                    num_generations=args.num_sample_generations,
                    max_new_tokens=args.max_new_tokens)

                # Append uncertainty: we use 1-int(correct) so that higher uncertainty corresponds to harmful responses
                uncertain_arr.append(1-int(correct))

                # For each key in entropies (if not excluded), append its value 
                for key, value in entropies.items():
                    if key not in excluded_keys:
                        scores_by_key[key].append(value[0])

            except Exception as e:
                logging.info(f"Error at index {idx}: {e}")
                continue

    elif args.dataset_name == "GMMLU":
        lang_codes = {"english": "en", "hindi": "hi", "bengali": "bn", "deutsch": "de"}
        Global_MMLU = load_dataset("CohereForAI/Global-MMLU", lang_codes[args.language.lower()])
        Global_MMLU.set_format("pandas")
        data = Global_MMLU['test'][:]
        # Filter data by subject_category
        category_data = data[data["subject_category"] == data_category]
        total_questions = category_data.shape[0]
        fraction = 2500 / total_questions
        actual_questions = 0
        # Get unique subjects within this category
        subjects = category_data["subject"].unique()

        for subject in subjects:
            subject_data = category_data[category_data["subject"] == subject].copy()
            length = subject_data.shape[0]
            length = int(length*fraction) + 1
        
            subject_data["total_length"] = (
                subject_data["question"].str.len() +
                subject_data["option_a"].str.len() +
                subject_data["option_b"].str.len() +
                subject_data["option_c"].str.len() +
                subject_data["option_d"].str.len()
            )

            # Find length of smallest two questions
            smallest_questions = subject_data.nsmallest(2, "total_length")

            # Find length of the first 2 questions in the subject
            length2 = smallest_questions["total_length"].sum()
            length3 = subject_data["total_length"].head(2).sum()

            # Build examples using the two smallest questions
            examples = []
            logging.info(f"Subject: {subject}")
            logging.info(f"Length2: {length2}")
            logging.info(f"Length3: {length3}")

            for i, (_, row) in enumerate(smallest_questions.iterrows()):
                examples.append({
                    "user": f"Q{i+1}: {row['question']}\nA. {row['option_a']}\nB. {row['option_b']}\nC. {row['option_c']}\nD. {row['option_d']}\nAnswer:\n\n",
                    "assistant": f"\n{row['answer']}\n\n"
                })

            # Select 'length' random questions excluding the ones in examples
            remaining_data = subject_data.drop(smallest_questions.index)
            random_questions = remaining_data.sample(n=length, random_state=args.seed_val).reset_index(drop=True)

            for i in tqdm(range(length), desc="Processing questions"):
                try:
                    actual_questions += 1
                    row = random_questions.iloc[i]
                    question = row['question']
                    if row['option_a'] == "" or row['option_b'] == "" or row['option_c'] == "" or row['option_d'] == "":
                        continue
                    choices = [row['option_a'], row['option_b'], row['option_c'], row['option_d']]
                    correct_choice = row['answer']
                    correct, entropies, generations = answer_question(
                        choices=choices, 
                        question=question, 
                        examples = examples,
                        correct_choice=correct_choice,
                        GMMLU_category=data_category,
                        tokenizer=tokenizer,
                        sbert_model=sbert_model,
                        model=model, 
                        device=device,
                        generations=generations, 
                        cutoff_score=args.cutoff_score, 
                        lang=args.language,
                        prompts_dir=args.prompts_dir,
                        dataset_name=args.dataset_name,
                        num_generations=args.num_sample_generations,
                        max_new_tokens=args.max_new_tokens)
                    # Append uncertainty: we use 1-int(correct) so that higher uncertainty corresponds to harmful responses
                    uncertain_arr.append(1-int(correct))

                    # For each key in entropies (if not excluded), append its value 
                    for key, value in entropies.items():
                        if key not in excluded_keys:
                            scores_by_key[key].append(value[0])
                
                except Exception as e:
                    logging.info(f"Error on index {i}: {e}")
                    continue

    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    compute_auroc(uncertain_arr, scores_by_key, output_file_path)
    saving_scores(uncertain_arr, output_file_path, generations, actual_questions, results_file_path)
    logging.info(f"Time taken: {(time.time()-time_now)/3600} hours for {data_category} category of {args.dataset_name} dataset with {args.model_name} model in {args.language} language.")


def main(args):
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name, hf_cache_dir="/data/hf_cache/", 
        max_memory=set_max_memory(args.gpu_index))
    sbert_model = load_sentence_transformer(model_name="sentence-transformers/LaBSE", device=device)

    if args.dataset_name == "mTREx":
        data_category_list = ["capitals", "country", "official_language"]
    elif args.dataset_name == "GMMLU":
        data_category_list = ["STEM", "Humanities"]
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}")
    
    for data_category in data_category_list:
        processing_dataset(data_category=data_category, 
                            model=model, 
                            tokenizer=tokenizer, 
                            sbert_model=sbert_model, 
                            device=device, 
                            args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Semantic Entropy method across different LLMs and languages for both mTREx and G-MMLU datasets.")
    parser.add_argument("--gpu_index", type=str, help="Give one aviable GPU index to utilize for inference", required=False, default="0")
    parser.add_argument("--model_name", type=str, choices=["Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct", "Mistral-7B-Instruct-v0.3", "Mistral-Small-24B-Instruct-2501"], help="Name of the model to use for inference", required=True)
    parser.add_argument("--language", type=str, choices=["English", "Hindi", "Bengali", "Deutsch", "Urdu"], help="Language to use for inference", required=True)
    parser.add_argument("--num_sample_generations", type=int, help="Number of sample generations to use for inference", required=False, default=20)
    parser.add_argument("--cutoff_score", type=float, help="Cutoff score for semantic similarity", required=False, default=0.75)
    parser.add_argument("--max_num_samples", type=int, help="Maximum number of samples to process from the dataset", required=False, default=2500)
    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of new tokens to generate", required=False, default=50)
    parser.add_argument("--mTREx_data_dir", type=str, help="Directory containing the mTREx dataset files", required=False, default="/data/debtanu/Research_work/low-resource-hallucination-detection/data/mTREx")
    parser.add_argument("--base_results_dir", type=str, help="Base directory to save results", required=False, default="/data/debtanu/Research_work")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use for inference", choices=["mTREx", "GMMLU"], required=True)
    parser.add_argument("--prompts_dir", type=str, help="Directory containing the prompts", required=False, default="/data/debtanu/Research_work/low-resource-hallucination-detection/prompts")
    parser.add_argument("--seed_val", type=int, help="Seed value for reproducibility", required=False, default=42)
    args = parser.parse_args()
    main(args)
