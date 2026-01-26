"""## Import necessary libraries and modules"""

import os, json, sys, gc, argparse, time, re, logging, torch, numpy as np, pandas as pd, random, pickle, sys, logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from collections import defaultdict
from functools import partial

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

def load_prompts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_tokenized_prompt(**token_prompt_args):
    tokenizer = token_prompt_args.get('tokenizer')
    messages = format_chat_prompt(**token_prompt_args)
    chat_template_message = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True, 
    )
    return chat_template_message

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
        
FULLY_CONNECTED_HIDDEN_LAYERS = defaultdict(list)

def save_fully_connected_hidden(name, mod, inp, out):
    FULLY_CONNECTED_HIDDEN_LAYERS[name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())

ATTENTION_HIDDEN_LAYERS = defaultdict(list)

def save_attention_hidden(name, mod, inp, out):
    ATTENTION_HIDDEN_LAYERS[name].append(out.squeeze().detach().to(torch.float32).cpu().numpy())

def collect_fully_connected(token_pos, activation_layer_name, total_layers):
    # Define middle and last layers
    if total_layers % 2 == 0:
        middle_layer = total_layers // 2 - 1  # For even number of layers, pick the lower middle layer
    else:
        middle_layer = total_layers // 2      # For odd number of layers, pick the exact middle layer
    
    last_layer = total_layers - 1

    # Collect activations for middle and last layers
    token_pos_activation = np.stack([
        FULLY_CONNECTED_HIDDEN_LAYERS[f'{activation_layer_name[0]}{middle_layer}{activation_layer_name[1]}'][-1][token_pos],
        FULLY_CONNECTED_HIDDEN_LAYERS[f'{activation_layer_name[0]}{last_layer}{activation_layer_name[1]}'][-1][token_pos]
    ])

    final_activation = np.stack([
        FULLY_CONNECTED_HIDDEN_LAYERS[f'{activation_layer_name[0]}{middle_layer}{activation_layer_name[1]}'][-1][-1],
        FULLY_CONNECTED_HIDDEN_LAYERS[f'{activation_layer_name[0]}{last_layer}{activation_layer_name[1]}'][-1][-1]
    ])

    return token_pos_activation, final_activation

def collect_fully_connected_token_pos_only(token_pos, activation_layer_name, total_layers):
    # Define middle and last layers
    if total_layers % 2 == 0:
        middle_layer = total_layers // 2 - 1  # For even number of layers, pick the lower middle layer
    else:
        middle_layer = total_layers // 2      # For odd number of layers, pick the exact middle layer
    
    last_layer = total_layers - 1

    # Collect activations for middle and last layers
    token_pos_activation = np.stack([
        FULLY_CONNECTED_HIDDEN_LAYERS[f'{activation_layer_name[0]}{middle_layer}{activation_layer_name[1]}'][-1][token_pos],
        FULLY_CONNECTED_HIDDEN_LAYERS[f'{activation_layer_name[0]}{last_layer}{activation_layer_name[1]}'][-1][token_pos]
    ])

    # final_activation = np.stack([
    #     FULLY_CONNECTED_HIDDEN_LAYERS[f'{activation_layer_name[0]}{middle_layer}{activation_layer_name[1]}'][-1][-1],
    #     FULLY_CONNECTED_HIDDEN_LAYERS[f'{activation_layer_name[0]}{last_layer}{activation_layer_name[1]}'][-1][-1]
    # ])

    return token_pos_activation

def collect_attention(token_pos, attention_layer_name, total_layers):
    # Define middle and last layers
    if total_layers % 2 == 0:
        middle_layer = total_layers // 2 - 1  # For even number of layers, pick the lower middle layer
    else:
        middle_layer = total_layers // 2      # For odd number of layers, pick the exact middle layer
    
    last_layer = total_layers - 1

    # Collect attention for middle and last layers
    token_pos_attention = np.stack([
        ATTENTION_HIDDEN_LAYERS[f'{attention_layer_name[0]}{middle_layer}{attention_layer_name[1]}'][-1][token_pos],
        ATTENTION_HIDDEN_LAYERS[f'{attention_layer_name[0]}{last_layer}{attention_layer_name[1]}'][-1][token_pos]
    ])

    final_attention = np.stack([
        ATTENTION_HIDDEN_LAYERS[f'{attention_layer_name[0]}{middle_layer}{attention_layer_name[1]}'][-1][-1],
        ATTENTION_HIDDEN_LAYERS[f'{attention_layer_name[0]}{last_layer}{attention_layer_name[1]}'][-1][-1]
    ])

    return token_pos_attention, final_attention

def collect_attention_token_pos_only(token_pos, attention_layer_name, total_layers):
    # Define middle and last layers
    if total_layers % 2 == 0:
        middle_layer = total_layers // 2 - 1  # For even number of layers, pick the lower middle layer
    else:
        middle_layer = total_layers // 2      # For odd number of layers, pick the exact middle layer
    
    last_layer = total_layers - 1

    # Collect attention for middle and last layers
    token_pos_attention = np.stack([
        ATTENTION_HIDDEN_LAYERS[f'{attention_layer_name[0]}{middle_layer}{attention_layer_name[1]}'][-1][token_pos],
        ATTENTION_HIDDEN_LAYERS[f'{attention_layer_name[0]}{last_layer}{attention_layer_name[1]}'][-1][token_pos]
    ])

    return token_pos_attention

def generate_response(**gen_res_args):
    max_new_tokens = gen_res_args.get('max_new_tokens')
    min_new_tokens = gen_res_args.get('min_new_tokens')
    tokenizer = gen_res_args.get('tokenizer')
    model = gen_res_args.get('model')
    device = gen_res_args.get('device')

    tokenized_inputs = tokenizer(
        text = gen_res_args.get('question'), 
        return_tensors="pt", 
        padding = True
    ).to(device)
    start_pos = tokenized_inputs["input_ids"].size(dim=-1)

    # We sample base response (zero temperature) for estimating model accuracy
    with torch.no_grad():
        model_outputs_1 = model.generate(
            input_ids=tokenized_inputs["input_ids"], attention_mask=tokenized_inputs["attention_mask"],
            do_sample=False,
            num_beams=1,
            temperature=0,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens, 
            return_dict_in_generate=True, 
            output_scores=True, 
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id
        )

    # Remove input prompt from generated output
    low_temp_response = tokenizer.decode(model_outputs_1.sequences[0][start_pos:], skip_special_tokens=True)
    low_temp_response = low_temp_response.replace("\n", " ").strip()

    with torch.no_grad():
        logits = model(model_outputs_1.sequences).logits.squeeze()
    
    del model_outputs_1
    gc.collect()
    
    return low_temp_response, logits, start_pos

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
        choices = ans_ques_args.get('choices')
        question = ans_ques_args.get('question')
        if len(choices) < 4: return 0
        formatted_question = (
            f"Q: {question}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}.\n"
        )
        prompts = load_prompts(os.path.join(ans_ques_args.get('prompts_dir'), f"{dataset_name}_prompts.json"))
        lang_prompts = prompts[lang]
        system_message = lang_prompts['system'].format(sub=ans_ques_args.get('GMMLU_category'))
        chat_str = get_tokenized_prompt(
                    system_message=system_message, 
                    actual_question=formatted_question,
                    examples = ans_ques_args.get('examples'),
                    tokenizer=tokenizer,
                    dataset_name=dataset_name)
        
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    str_response, logits, start_pos = generate_response(
        question=chat_str, 
        max_new_tokens=ans_ques_args.get('max_new_tokens'),
        min_new_tokens=ans_ques_args.get('min_new_tokens'),
        tokenizer=tokenizer,
        model=model,
        device=device, 
    )

    if dataset_name == "mTREx":
        target_phrases, correct = post_process_mtrex_response(target=target, str_response=str_response, user_message=user_message)
        final_target = target_phrases
    elif dataset_name == "GMMLU":
        final_target = ans_ques_args.get('correct_choice')
        decoded_tokens, correct = post_process_GMMLU_response(correct_choice=final_target, str_response=str_response)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    result = {
            'question': chat_str,
            'target': final_target,
            'start_pos': start_pos,
            'response': str_response,
            'correct': correct,
        }

    generations.append(result)

    return correct, start_pos, logits, generations, str_response

def saving_scores(**save_args):
    dataset_name = save_args.get('dataset_name')
    actual_questions = save_args.get('actual_questions')
    generations = save_args.get('generations')
    output_file_path = save_args.get('output_file_path')
    results_file_path = save_args.get('results_file_path')

    if dataset_name == "mTREx":
        num_correct_1 = save_args.get('num_correct_1')
        num_correct_2 = save_args.get('num_correct_2')
        logging.info(f"Number of true points: {num_correct_1}")
        logging.info(f"Number of true points (without '?'): {num_correct_2}")
        logging.info(f"Total questions processed: {actual_questions}")
        logging.info(f"Accuracy (with '?'): {num_correct_1/actual_questions}")
        logging.info(f"Accuracy (without '?'): {num_correct_2/actual_questions}")
        with open(output_file_path,'a', encoding='utf-8') as f:
            print(f"Number of true points: {num_correct_1}", file=f)
            print(f"Number of true points (without '?'): {num_correct_2}", file=f)
            print(f"Total questions processed: {actual_questions}", file=f)
            print(f"Accuracy (with '?'): {num_correct_1/actual_questions}", file=f)
            print(f"Accuracy (without '?'): {num_correct_2/actual_questions}", file=f)

    elif dataset_name == "GMMLU":
        correct_count = save_args.get('num_correct')
        logging.info(f"Number of true points: {correct_count}")
        logging.info(f"Total questions processed: {actual_questions}")
        logging.info(f"Accuracy: {correct_count/actual_questions}")

        with open(output_file_path,'a', encoding='utf-8') as f:
            print(f"Number of true points: {correct_count}", file=f)
            print(f"Total questions processed: {actual_questions}", file=f)
            print(f"Accuracy: {correct_count/actual_questions}", file=f)

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Supported datasets are 'mTREx' and 'GMMLU'.")
    
    with open(results_file_path, 'w', encoding='utf-8') as f:
        json.dump(generations, f, ensure_ascii=False, indent=4)
    logging.info(f"Results saved to {results_file_path}")

def load_dataset_mTREx(**data_mtrex_args):
    dataset_df = pd.read_csv(f"{data_mtrex_args.get('data_dir')}/{data_mtrex_args.get('lang')}/{data_mtrex_args.get('mTREx_category')}.csv")
    data = []
    for idx in range(data_mtrex_args.get('max_num_samples')):
        data.append((dataset_df.iloc[idx]['source'], dataset_df.iloc[idx]['target']))
        if idx >= len(dataset_df) - 1:
            break
    final_data = []
    for src, tgt in data:
        final_data.append((src, tgt[0:]))
    return final_data


def processing_dataset(data_category, model, tokenizer, device, coll_str, args):
    results_dir = f"{args.base_results_dir}/ModelArti_{args.dataset_name}_results_v1/{args.model_name}_ModelArti_{data_category}_{args.language}_results"
    os.makedirs(results_dir, exist_ok=True)

    results_pickle_dir = os.path.join(results_dir, "pickle_files")
    os.makedirs(results_pickle_dir, exist_ok=True)

    results_file_path = os.path.join(results_dir, f"{args.dataset_name}_{args.model_name}_{data_category}_{args.language}.json")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                    logging.FileHandler(f"{results_dir}/{args.model_name}_ModelArti_{data_category}_{args.language}_results_log.txt"),
                    logging.StreamHandler(sys.stdout)
                ])
    
    logging.info(f"Using model: {args.model_name}, language: {args.language}, dataset: {args.dataset_name}, category: {data_category}")

    output_file_path = os.path.join(results_dir, f'{args.dataset_name}_{args.model_name}_{data_category}_{args.language}.txt')
    with open(output_file_path,'w', encoding='utf-8') as f:
        logging.info(f"Evaluating {data_category} category")
        print(f"Evaluating {data_category} category", file=f)

    model_repos = {
        "Llama-3.1-8B-Instruct": ("meta-llama", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
        "Llama-3.3-70B-Instruct": ("meta-llama", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
        "Mistral-7B-Instruct-v0.3": ("mistralai", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
        "Mistral-Small-24B-Instruct-2501": ("mistralai", f".*model.layers.{coll_str}.mlp.up_proj", f".*model.layers.{coll_str}.self_attn.o_proj"),
    }

    FULLY_CONNECTED_FORWARD_HANDLES = {}

    for name, module in model.named_modules():
        if re.match(f'{model_repos[args.model_name][1]}$', name):
            FULLY_CONNECTED_FORWARD_HANDLES[name] = module.register_forward_hook(partial(save_fully_connected_hidden, name))

    ATTENTION_FORWARD_HANDLES = {}

    for name, module in model.named_modules():
        if re.match(f'{model_repos[args.model_name][2]}$', name):
            ATTENTION_FORWARD_HANDLES[name] = module.register_forward_hook(partial(save_attention_hidden, name))

    activation_layer_name = model_repos[args.model_name][1][2:].split(coll_str)
    attention_layer_name = model_repos[args.model_name][2][2:].split(coll_str)
    total_model_layers = len(model.model.layers)

    generations = []
    time_now = time.time()

    lang_codes = {"english": "en", "hindi": "hi", "bengali": "bn", "deutsch": "de", "urdu": "ur"}

    if args.dataset_name == "mTREx":
        actual_questions = 0
        data = load_dataset_mTREx(
            data_dir = args.mTREx_data_dir, 
            lang = args.language, 
            mTREx_category = data_category, 
            max_num_samples = args.max_num_samples,
            seed_val = args.seed_val)
        
        chunk_size = 1
        chunk_data = []
        num_correct_1, num_correct_2 = 0, 0

        for idx in tqdm(range(len(data)), desc="Processing mTREx questions"):
            try:
                FULLY_CONNECTED_HIDDEN_LAYERS.clear()
                ATTENTION_HIDDEN_LAYERS.clear()

                actual_questions += 1
                source, target = data[idx]

                correct, start_pos, logits, generations, str_response = answer_question(
                        source=source,
                        target=target,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        generations=generations,
                        mTREx_category=data_category,
                        lang=args.language,
                        prompts_dir=args.prompts_dir,
                        dataset_name=args.dataset_name,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.min_new_tokens
                )

                if correct == 1:
                    num_correct_1 += 1
                    if '?' not in str_response:
                        num_correct_2 += 1

                first_fully_connected, final_fully_connected = collect_fully_connected(start_pos, activation_layer_name, total_model_layers)

                token_pos_fully_connected_list = []
                token_cnt_tmp = 0
                while token_cnt_tmp < args.avg_num_tokens:
                    token_pos = start_pos + token_cnt_tmp
                    try:
                        token_pos_vec = collect_fully_connected_token_pos_only(token_pos, activation_layer_name, total_model_layers)
                        token_pos_fully_connected_list.append(token_pos_vec)
                        token_cnt_tmp += 1
                    except (KeyError, IndexError):
                        # no more tokens available
                        break
                token_pos_fully_connected_list = np.stack(token_pos_fully_connected_list)
                token_pos_fully_connected_avg = np.mean(token_pos_fully_connected_list, axis=0)

                assert token_pos_fully_connected_avg.shape == first_fully_connected.shape, f"Shape mismatch: average shape {token_pos_fully_connected_avg.shape}, first shape {first_fully_connected.shape}"
                assert token_pos_fully_connected_avg.shape == final_fully_connected.shape, f"Shape mismatch: average shape {token_pos_fully_connected_avg.shape}, final shape {final_fully_connected.shape}"

                first_attention, final_attention = collect_attention(start_pos, attention_layer_name, total_model_layers)

                token_pos_attention_list = []
                token_cnt_tmp = 0
                while token_cnt_tmp < args.avg_num_tokens:
                    token_pos = start_pos + token_cnt_tmp
                    try:
                        token_pos_vec = collect_attention_token_pos_only(token_pos, attention_layer_name, total_model_layers)
                        token_pos_attention_list.append(token_pos_vec)
                        token_cnt_tmp += 1
                    except (KeyError, IndexError):
                        # no more tokens available
                        break
                token_pos_attention_list = np.stack(token_pos_attention_list)
                token_pos_attention_avg = np.mean(token_pos_attention_list, axis=0)

                assert token_pos_attention_avg.shape == first_attention.shape, f"Shape mismatch: average shape {token_pos_attention_avg.shape}, first shape {first_attention.shape}"
                assert token_pos_attention_avg.shape == final_attention.shape, f"Shape mismatch: average shape {token_pos_attention_avg.shape}, final shape {final_attention.shape}"

                first_logits = logits[start_pos].to(torch.float32).cpu().numpy()

                result_entry = {
                    "source": source,
                    "target": target,
                    "str_response": str_response,
                    "correct": correct,
                    "first_logits": first_logits,
                    "start_pos": start_pos,
                    "first_fully_connected": first_fully_connected,
                    # "final_fully_connected": final_fully_connected,
                    "token_pos_fully_connected_avg": token_pos_fully_connected_avg,
                    "first_attention": first_attention,
                    # "final_attention": final_attention,
                    "token_pos_attention_avg": token_pos_attention_avg,
                }

                chunk_data.append(result_entry)

            except Exception as e:
                logging.info(f"Error on index {idx}: {e}")
                continue
            
            # Save chunk data to pickle file
            if (idx + 1) % chunk_size == 0 or (idx + 1) == len(data):
                chunk_file_path = os.path.join(results_pickle_dir, f"{data_category}_chunk_{(idx + 1) // chunk_size}.pickle")
                with open(chunk_file_path, 'wb') as chunk_file:
                    for entry in chunk_data:
                        pickle.dump(entry, chunk_file)
                chunk_data.clear()
        
        saving_scores(
            num_correct_1 = num_correct_1,
            num_correct_2 = num_correct_2,
            output_file_path = output_file_path,
            generations = generations,
            actual_questions = actual_questions,
            results_file_path = results_file_path, 
            dataset_name = args.dataset_name,
        )

    elif args.dataset_name == "GMMLU":
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

        num_correct = 0

        for subject in subjects:
            subject_data = category_data[category_data["subject"] == subject].copy()
            length = subject_data.shape[0]
            length = int(length * fraction)+1
            
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

            chunk_data = []

            for i in tqdm(range(length), desc="Processing questions"):
                try:
                    FULLY_CONNECTED_HIDDEN_LAYERS.clear()
                    ATTENTION_HIDDEN_LAYERS.clear()

                    actual_questions += 1
                    row = random_questions.iloc[i]
                    question = row['question']
                    if row['option_a'] == "" or row['option_b'] == "" or row['option_c'] == "" or row['option_d'] == "":
                        continue
                    choices = [row['option_a'], row['option_b'], row['option_c'], row['option_d']]
                    correct_choice = row['answer']

                    correct, start_pos, logits, generations, str_response = answer_question(
                        choices=choices,
                        question=question,
                        examples=examples,
                        correct_choice=correct_choice,
                        GMMLU_category=data_category,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        generations=generations,
                        lang=args.language,
                        prompts_dir=args.prompts_dir,
                        dataset_name=args.dataset_name,
                        max_new_tokens=args.max_new_tokens,
                        min_new_tokens=args.min_new_tokens
                    )

                    first_fully_connected, final_fully_connected = collect_fully_connected(start_pos, activation_layer_name, total_model_layers)

                    token_pos_fully_connected_list = []
                    token_cnt_tmp = 0
                    while token_cnt_tmp < args.avg_num_tokens:
                        token_pos = start_pos + token_cnt_tmp
                        try:
                            token_pos_vec = collect_fully_connected_token_pos_only(token_pos, activation_layer_name, total_model_layers)
                            token_pos_fully_connected_list.append(token_pos_vec)
                            token_cnt_tmp += 1
                        except (KeyError, IndexError):
                            # no more tokens available
                            break
                    token_pos_fully_connected_list = np.stack(token_pos_fully_connected_list)
                    token_pos_fully_connected_avg = np.mean(token_pos_fully_connected_list, axis=0)

                    assert token_pos_fully_connected_avg.shape == first_fully_connected.shape, f"Shape mismatch: average shape {token_pos_fully_connected_avg.shape}, first shape {first_fully_connected.shape}"
                    assert token_pos_fully_connected_avg.shape == final_fully_connected.shape, f"Shape mismatch: average shape {token_pos_fully_connected_avg.shape}, final shape {final_fully_connected.shape}"

                    first_attention, final_attention = collect_attention(start_pos, attention_layer_name, total_model_layers)

                    token_pos_attention_list = []
                    token_cnt_tmp = 0
                    while token_cnt_tmp < args.avg_num_tokens:
                        token_pos = start_pos + token_cnt_tmp
                        try:
                            token_pos_vec = collect_attention_token_pos_only(token_pos, attention_layer_name, total_model_layers)
                            token_pos_attention_list.append(token_pos_vec)
                            token_cnt_tmp += 1
                        except (KeyError, IndexError):
                            # no more tokens available
                            break
                    token_pos_attention_list = np.stack(token_pos_attention_list)
                    token_pos_attention_avg = np.mean(token_pos_attention_list, axis=0)

                    assert token_pos_attention_avg.shape == first_attention.shape, f"Shape mismatch: average shape {token_pos_attention_avg.shape}, first shape {first_attention.shape}"
                    assert token_pos_attention_avg.shape == final_attention.shape, f"Shape mismatch: average shape {token_pos_attention_avg.shape}, final shape {final_attention.shape}"

                    if correct == 1: num_correct += 1

                    first_logits = logits[start_pos].to(torch.float32).cpu().numpy()

                    result_entry = {
                        "question": question,
                        "correct_choice": correct_choice,
                        "subject": subject,
                        "subject_category": data_category,
                        "str_response": str_response,
                        "correct": correct,
                        "first_logits": first_logits,
                        "start_pos": start_pos,
                        "first_fully_connected": first_fully_connected,
                        # "final_fully_connected": final_fully_connected,
                        "token_pos_fully_connected_avg": token_pos_fully_connected_avg,
                        "first_attention": first_attention,
                        # "final_attention": final_attention,
                        "token_pos_attention_avg": token_pos_attention_avg,
                    }

                    chunk_data.append(result_entry)

                except Exception as e:
                    logging.info(f"Error on index {i}: {e}")
                    continue
            
            # Save chunk data to pickle file
            chunk_file_path = os.path.join(results_pickle_dir, f"{data_category}_{subject}_chunk.pickle")
            with open(chunk_file_path, 'wb') as chunk_file:
                for entry in chunk_data:
                    pickle.dump(entry, chunk_file)

        saving_scores(
            num_correct = num_correct,
            output_file_path = output_file_path,
            generations = generations,
            actual_questions = actual_questions,
            results_file_path = results_file_path,
            dataset_name = args.dataset_name,
        )

    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}. Supported datasets are 'mTREx' and 'GMMLU'.")
    
    
    logging.info(f"Time taken: {(time.time()-time_now)/3600} hours for {data_category} category of {args.dataset_name} dataset with {args.model_name} model in {args.language} language.")

def main(args):
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name, hf_cache_dir="/data/hf_cache/", 
        max_memory=set_max_memory(args.gpu_index))
    
    model_num_layers = {
        "Llama-3.1-8B-Instruct" : 32,
        "Llama-3.3-70B-Instruct" : 80,
        "Mistral-7B-Instruct-v0.3" : 32,
        "Mistral-Small-24B-Instruct-2501" : 40
    }

    assert args.layer_number < model_num_layers[args.model_name], f"Layer number {args.layer_number} is out of range for model {args.model_name} with {model_num_layers[args.model_name]} layers."

    coll_str = "[0-9]+" if args.layer_number == -1 else str(args.layer_number)

    if args.dataset_name == "mTREx":
        data_category_list = ["capitals", "country", "official_language"]
    elif args.dataset_name == "GMMLU":
        data_category_list = ["STEM", "Humanities"]
    else:
        raise ValueError(f"Unsupported dataset name: {args.dataset_name}. Supported datasets are 'mTREx' and 'GMMLU'.")
    
    for data_category in data_category_list:
        processing_dataset(
            data_category=data_category, 
            model=model, 
            tokenizer=tokenizer, 
            device=device,
            coll_str=coll_str,
            args=args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model Artifacts method across different LLMs and languages for G-MMLU and mTREx datasets.")
    parser.add_argument("--gpu_index", type=str, help="Give one aviable GPU index to utilize for inference", required=False, default="0")
    parser.add_argument("--model_name", type=str, choices=["Llama-3.1-8B-Instruct", "Llama-3.3-70B-Instruct", "Mistral-7B-Instruct-v0.3", "Mistral-Small-24B-Instruct-2501"], help="Name of the model to use for inference", required=True)
    parser.add_argument("--layer_number", type=int, help="Layer number to use for inference. -1 means all layers", required=False, default=-1)
    parser.add_argument("--language", type=str, choices=["English", "Hindi", "Bengali", "Deutsch", "Urdu"], help="Language to use for inference", required=True)
    parser.add_argument("--max_num_samples", type=int, help="Maximum number of samples to process from the dataset", required=False, default=2500)
    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of new tokens to generate", required=False, default=50)
    parser.add_argument("--min_new_tokens", type=int, help="Minimum number of new tokens to generate", required=False, default=5)
    parser.add_argument("--mTREx_data_dir", type=str, help="Directory containing the mTREx dataset files", required=False, default="/data/debtanu/Research_work/low-resource-hallucination-detection/data/mTREx")
    parser.add_argument("--base_results_dir", type=str, help="Base directory to save results", required=False, default="/data/debtanu/Research_work/Results_ModelArti_Rebuttal")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use for inference", choices=["mTREx", "GMMLU"], required=True)
    parser.add_argument("--prompts_dir", type=str, help="Directory containing the prompts", required=False, default="/data/debtanu/Research_work/low-resource-hallucination-detection/prompts")
    parser.add_argument("--seed_val", type=int, help="Seed value for reproducibility", required=False, default=42)
    parser.add_argument("--avg_num_tokens", type=int, help="Number of tokens to average over for token position representations", required=False, default=10)
    args = parser.parse_args()
    main(args)

