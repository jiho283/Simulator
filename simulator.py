import os
import pickle
from copy import deepcopy
import time
import random
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
import time
import signal
import warnings
import argparse
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

from logging_results.logging import log_results, log_everything
from post_processing.process_answer import judge_eq, distill_answer, calibrate
from models.api_based_inference import gpt_inference, claude_inference, gemini_inference
from models.open_source_model_inference import open_source_model_inference
from models.load_opensource_model import load_opensource_tokenizer
from models.load_model import load_model
from utils.utils import get_embedding, search_history, open_file, name_change, extract_gt_sessions_bm25_date
warnings.filterwarnings('ignore')
from func_timeout import func_set_timeout, FunctionTimedOut, func_timeout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", help="name of the model. Default: 'gpt-3.5-turbo'.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="if set, use truncated dataset for debugging.")
    parser.add_argument("--debug_n_episodes", type=int, default=5, help="number of episodes to evalutate on debug mode.")
    parser.add_argument("--quantization", type=str, default="no", help="either to quantize the model or not. Default: False")
    parser.add_argument("--script_name", type=str, default='friends', help="name of the script to evaluate. Should be one of ('friends', 'bigbang', 'theoffice'). Default: 'friends'")
    parser.add_argument("--sleep_time", type=float, default=5, help="time limit in seconds for model response. Default: 5")
    parser.add_argument('--history_type', type=str, default='session-entire', help="How to store conversation history.")
    parser.add_argument('--num_ret_history', type=int, default=10, help="Number of histories we are going to retrieve. Default: 10.")
    parser.add_argument('--ret_method', type=str, default='bm25', help=" Default: openai-emb. Should be one of ('openai-emb', 'bm25', 'no_ret')")
    parser.add_argument('--name_shuffle', type=str, default='original', help=" Default: original. Should be one of ('original', 'shuffle', 'new_name')")
    parser.add_argument('--trial_version', type=int, default=0, help= "version number of the experiment.")
    parser.add_argument('--sh_number', type=int, default=0, help='shell script number')
    parser.add_argument('--num_cores', type=int, default=10, help='upper bound of number of cpu cores')
    parser.add_argument('--openai_api_key', type=str, default="", help="OpenAI API Key")
    parser.add_argument('--gemini_api_key', type=str, default="", help="Gemini API key")
    parser.add_argument('--antrhopic_api_key', type=str, default="", help="Anthropic API key")
    parser.add_argument('--fast_eval', type=str, default="yes", help="When set to 'yes', the simulator proceeds to the next utterance without waiting for the time interval if the history has already been updated. Should be one of ('yes', 'no')")
    parser.add_argument('--answer_format', type=str, default='multi_choice_structured', help="the format of the answer of the agent.")
    return parser.parse_args()

def answer_question(model_name, client, model, tokenizer, config, prompt):
    answer = ""
    try:
        if "gpt" in model_name.lower():
            answer = gpt_inference(prompt, model_name, client)
        elif model_name == "claude-3" or model_name == "claude-2.1":
            answer = claude_inference(prompt, model_name, client)
        elif model_name == "gemini":
            answer = gemini_inference(prompt, model)
        else:
            answer = open_source_model_inference(prompt, model_name, model, tokenizer, config)
    except:
        pass
    return answer

def retrieve_history(ret_method, num_ret_history, openai_client, max_token_len, save_result, char_ask_sh, real_question_sh, data_dict, gt_sessions):
    ret_histories = ''
    if ret_method == 'openai-emb': 
        if len(data_dict['history']) == 0:
            ret_histories = "No history.\n"
        else:
            res = search_history(save_result, f'{char_ask_sh}: {real_question_sh}', client=openai_client, n=num_ret_history)     
            for ret_history in list(res['history']):
                ret_histories = ret_histories + ret_history + '\n'
    elif ret_method == 'bm25':
        if len(data_dict['history']) == 0:
            ret_histories = "No history.\n"
        else:
            tokenized_query = word_tokenize(f'{char_ask_sh}: {real_question_sh}'.lower())
            doc_scores = save_result.get_scores(tokenized_query)
            top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:num_ret_history]
            top_docs = [data_dict['history'][i] for i in top_doc_indices]
            for ret_history in top_docs:
                ret_histories = ret_histories + ret_history + '\n'
    elif ret_method == 'no_ret':
        total_token_len = 0
        ret_his_inds = []
        if len(data_dict['history']) == 0:
            ret_histories = "No history.\n"
        else:
            for h_ind in range(len(data_dict['ada_embedding'])):
                total_token_len += data_dict['ada_embedding'][-1-h_ind]
                if total_token_len > max_token_len - 500:
                    break
                ret_his_inds.append(-1-h_ind)
                ret_histories =  data_dict['history'][-1-h_ind] + '\n' + ret_histories    
    elif ret_method == 'oracle':
        ret_histories = gt_sessions                                    
    return ret_histories


def save_history(history_num, history, history_type, date, cur_conv_num, un, post_utterances, utter_post, model_name, client, model, tokenizer, config, data_dict, ret_method, llama_tokenizer):     
    if history_type == "utts":
        processed_history = f"[Date: {date}, Session #{cur_conv_num}, Utterance #{history_num+1}] {history}"
        history_num += 1
    elif history_type == "session-entire":
        processed_history = f"[Date: {date}, Session #{cur_conv_num}]\n{history}"
    elif history_type == "session-summary":
        history_sum = ""
        if un == len(post_utterances)-1:
            sum_prompt = open_file('./prompt/chatgpt_summarize_prompt.txt').replace('<<<DIALOG>>>', history)
            try:
                if "gpt" in model_name.lower():
                    history_sum = gpt_inference(sum_prompt, model_name, client)
                elif model_name == "claude-3" or model_name == "claude-2.1":
                    history_sum = claude_inference(sum_prompt, model_name, client)
                elif model_name == "gemini":
                    history_sum = gemini_inference(sum_prompt, model)
                else:
                    history_sum = open_source_model_inference(sum_prompt, model_name, model, tokenizer, config)
            except:
                pass
        else:
            history_sum = history
        processed_history = f"[Date: {date}, Session #{cur_conv_num}]\n{history_sum}\n"

    data_dict['history'].append(processed_history)
    if ret_method == 'openai-emb':
        embedding_vec = get_embedding(processed_history, client=client, model="text-embedding-3-small")
        data_dict['ada_embedding'].append(embedding_vec)
        data_df = pd.DataFrame(data_dict)
        return data_df, history_num#, data_dict
    elif ret_method == 'bm25':
        tokenized_docs = [word_tokenize(doc.lower()) for doc in data_dict['history']]
        bm25 = BM25Okapi(tokenized_docs)
        return bm25, history_num#, data_dict
    elif ret_method == 'no_ret':
        token_len = llama_tokenizer(processed_history, return_tensors="pt", truncation=True).input_ids.shape[1]
        data_dict['ada_embedding'].append(token_len)
        return None, history_num#, data_dict
    elif ret_method == "oracle":
        return None, history_num#, data_dict
    else:
        return AssertionError("Incorrect `ret_method`.")
        

def simulator(
        script_name,
        sleep_time=5, 
        tkg_ratio=0.7, 
        num_ret_history = 5, 
        model_name:str="gpt-3.5-turbo", 
        debug:bool=False, 
        debug_n_episodes:int=5,
        quantization:str="no",
        history_type:str="session-entire",
        ret_method:str='bm25',
        name_shuffle:str='original',
        openai_api_key:str="",
        gemini_api_key:str="",
        antrhopic_api_key:str="",
        fast_eval:str="yes",
        answer_format:str="multi_choice_structured"
        ):
    """
    script_name: script name ('friends', 'bigbang', 'theoffice')
    sleep_time: time for one utterance in the simulator. we do not use this parameter in unlimited simulator. (e.g. 3)
    tkg_ratio: the ratio of KG based question among the whole question set (0.0-1.0)
    mode: question type ('select' or 'free text')
    num_ret_history: the number of the retrieved utterances
    ret_method: retrieval method. openai embedding based: 'openai-emb', BM25 based: 'bm25', Naive LLM inference: 'no_ret'.
    """
    
    model, tokenizer, config = load_model(model_name, quantization, gemini_api_key=gemini_api_key)
    simulator_start_time = time.time()
    if ret_method in ['no_ret', 'oracle']:
        history_type = 'session-entire'
    
    ####For hyperparameter 
    if history_type == "utts":
        num_ret_history = 20
    elif history_type == "session-entire":
        if 'llama' in model_name.lower():
            num_ret_history = 3
            if ret_method == 'bm25':
                num_ret_history = 1
        elif 'tulu' in model_name.lower() or 'gemma' in model_name.lower():
            if ret_method == 'bm25':
                num_ret_history = 2
            else:
                num_ret_history = 5
        else:
            num_ret_history = 10
    elif history_type == "session-summary":
        if 'llama' in model_name.lower():
            num_ret_history = 8
        else:
            num_ret_history = 15
    
    if ret_method == 'no_ret':
        llama_tokenizer = load_opensource_tokenizer("llama2-7b-chat")
    else:
        llama_tokenizer = ""

    max_token_len = 0
    if model_name == "gpt-3.5-turbo":
        max_token_len = 16000
    elif "gpt-4" in model_name.lower():
        max_token_len = 128000
    elif model_name == "claude-3" or model_name == "claude-2.1":
        max_token_len = 200000
    elif model_name == "gemini":
        max_token_len = 32000
    elif 'tulu' in model_name.lower():
        max_token_len = 6000
    else:
        try:
            max_token_len = config.max_position_embeddings
        except:
            max_token_len = 4000
    
    if ret_method == "oracle":
        if "gpt" in model_name.lower():
            num_ret_history = 20
        elif model_name == "claude-3" or model_name == "claude-2.1":
            num_ret_history = 20
        elif "gemini" in model_name.lower():
            num_ret_history = 20
        elif 'tulu' in model_name.lower():
            num_ret_history = 4
        elif 'llama2' in model_name.lower():
            num_ret_history = 2
        elif 'gemma' in model_name.lower():
            num_ret_history = 10
        else:
            num_ret_history = 20
    client = None
    openai_client = None
    if 'gpt' in model_name.lower() or 'openai' in ret_method:
        openai_client = OpenAI(api_key=openai_api_key) 
        
    elif answer_format in ['multi_choice_unstructured', 'open_ended']:
        openai_client = OpenAI(api_key=openai_api_key)
        if "claude" not in model_name.lower() or "gemini" not in model_name.lower():
            client = openai_client
    anthropic_client = None
    if "claude" in model_name.lower(): 
        anthropic_client = Anthropic(api_key=antrhopic_api_key)
    
    elif "gpt" in model_name.lower():
        client = openai_client
        if answer_format in ['multi_choice_unstructured', 'open_ended']:
            client = OpenAI(api_key=openai_api_key)
    elif "claude" in model_name.lower():
        client = anthropic_client
    
    with open(f'./data/{script_name}_dialsim.pickle', 'rb') as f:
        data = pickle.load(f)
    with open(f'./data/{script_name}_oracle_tkg.pickle', 'rb') as f_h:
        oracle_tkg = pickle.load(f_h)
    with open(f'./data/{script_name}_oracle_fan.pickle', 'rb') as f_e:
        oracle_fan = pickle.load(f_e)

    if script_name == 'friends':
        chatbot = 'Ross'
    elif script_name == 'bigbang':
        chatbot = 'Sheldon'
    elif script_name == 'theoffice':
        chatbot = 'Michael'
    else:
        assert 0
    
    data_dict = {
        'ada_embedding': [], ### openai-emb -> embedding vector, no_ret -> token length
        'history': []
    }

    episodes = list(data)
    if debug:
        episodes = episodes[:debug_n_episodes]
    before_date = ''
    cur_conv_num = 1

    result_list = []
    result_time_list = []
    ambiguous_idx_list = [] # list of indices of the data (episode, session, question_prompt) where the model's output is ambiguous. 
    ambiguous_answer_list = [] # list of answers(model output) that are ambiguous.
    ambiguous_gold_answer_list = [] # list of ground truth answers for the ambiguous answers.
    answer_list = [] # list of answers generated by the models. TODO: implement logging answers too.
    gold_answer_list = [] # list of ground truth (gold) answers
    ret_histories_question_answer_list = [] # list of (ret_histories, question)
    save_time_list = [] # list of saving time
    retrieve_search_time_list = [] # list of time spent in `search_history`
    ans_time_list = [] # list of time spent in answering
    calibrated_result_list = [] # list of calibrated answers
    calibrated_distilled_answer_list = [] # list of calibrated distilled answers
    epi_session_date_to_sessions = {} 
    date_to_sessions = {}
    target_level_list = []

    for epi in episodes:
        epi_session_date_to_sessions[epi] = {}
        epi_data = data[epi]
        session_nums = list(epi_data)
        
        for sc_num in session_nums:
            already_asked = 0
            script = epi_data[sc_num]['script']
            date = epi_data[sc_num]['date']
            date_splitted = date.replace(',', '').split()
            cannot_tkg = 0
            cannot_fan = 0
            temp_script = name_change(script_name, script, name_shuffle)
            epi_session_date_to_sessions[epi][sc_num] = {date: temp_script}
            
            try:
                date_to_sessions[date].append(temp_script)
            except:
                date_to_sessions[date] = [temp_script]

            ###Whether it is possible to ask tkg-based questions 
            try:
                question_dict = epi_data[sc_num]['hard_q']
                final_tkg_list = []
                tkg_list = list(question_dict)
                for tkg in tkg_list:
                    if len(question_dict[tkg]) > 0:
                        final_tkg_list.append(tkg)
                tkg_target_type = random.choice(final_tkg_list)

                tkg_q_list = question_dict[tkg_target_type]
                target_question = random.choice(tkg_q_list)
            except:
                cannot_tkg=1
                pass

            ###Whether it is possible to ask fan quiz-based questions 
            try:
                question_dict = epi_data[sc_num]['easy_q']
                final_fan_list = []
                fan_list = list(question_dict)
                for fan in fan_list:
                    if len(list(question_dict[fan])) > 0:
                        final_fan_list.append(fan)
                fan_target_type = random.choice(final_fan_list)

                fan_q_list = list(question_dict[fan_target_type])
                fan_q_target_num = random.choice(fan_q_list)
                target_question = question_dict[fan_target_type][fan_q_target_num]
            except:
                cannot_fan = 1
                pass

            target_question_list = []
            current_type = ''
            gt_sessions = ""
            target_dates_list = []

            #### Question Selection (tkg or fan)
            rand_val = random.random()
            if cannot_fan == 1 and cannot_tkg == 1:
                target_question_list = ['cannot ask' for _ in range(20)]
            elif (cannot_fan == 1 and cannot_tkg == 0) or rand_val < tkg_ratio:
                question_dict = epi_data[sc_num]['hard_q']
                final_tkg_list = []
                fu_num = 0
                not_fu_list = []
                tkg_list = list(question_dict)
                for tkg in tkg_list:
                    if len(question_dict[tkg]) > 0:
                        final_tkg_list.append(tkg)
                        if 'fu' in tkg:
                            fu_num += 1
                        else:
                            not_fu_list.append(tkg)
                if len(not_fu_list) > 0:
                    random.shuffle(not_fu_list)
                    while True:
                        should_stop = 0
                        for not_fu in not_fu_list:
                            if fu_num/len(final_tkg_list) < 0.215:
                                should_stop = 1
                                break
                            final_tkg_list.append(not_fu)
                        if should_stop == 1:
                            break
                tkg_target_type = random.choice(final_tkg_list)
                tkg_q_list = question_dict[tkg_target_type]

                current_type = tkg_target_type
                for _ in range(20):
                    target_question = random.choice(tkg_q_list)
                    ran_q = target_question['questions'][list(target_question['questions'])[0]]
                    if 'n '+ date_splitted[2] in ran_q or date_splitted[0] + ' ' + date_splitted[2] in ran_q:
                        continue
                    final_target_question = deepcopy(target_question)
                    target_question_list.append(final_target_question)
                    
                    try:
                        target_dates_list.append(oracle_tkg[epi][sc_num][current_type][tkg_q_list.index(target_question)])
                    except:
                        try:
                            target_dates_list.append(oracle_tkg[epi][sc_num][current_type][target_question['questions'][list(target_question['questions'])[0]]])
                        except:
                            target_dates_list.append([])

            elif (cannot_fan == 0 and cannot_tkg == 1) or rand_val >= tkg_ratio:
                question_dict = epi_data[sc_num]['easy_q']
                final_fan_list = []
                unans_num = 0
                ans_list = []
                fan_list = list(question_dict)
                for fan in fan_list:
                    if len(list(question_dict[fan])) > 0:
                        final_fan_list.append(fan)
                        if 'unans' in fan:
                            unans_num += 1
                        else:
                            ans_list.append(fan)
                
                if len(ans_list) > 0:
                    random.shuffle(ans_list)
                    while True:
                        should_stop = 0
                        for ans_ele in ans_list:
                            if unans_num/len(final_fan_list) < 0.27:
                                should_stop = 1
                                break
                            final_fan_list.append(ans_ele)
                        if should_stop == 1:
                            break

                fan_target_type = random.choice(final_fan_list)
                fan_q_list = list(question_dict[fan_target_type]) 
                current_type = fan_target_type
            
                for _ in range(20):
                    fan_q_target_num = random.choice(fan_q_list) 
                    target_question = deepcopy(question_dict[fan_target_type][fan_q_target_num])
                    target_question_list.append(target_question)
                    if current_type in ['ans_w_time', 'dont_know_unans_time']:
                        try:
                            target_dates_list.append(oracle_fan[epi][sc_num][current_type][fan_q_target_num])
                        except:
                            target_dates_list.append([])
                    else:
                        target_dates_list.append([])

            if before_date != date:
                cur_conv_num = 1
                before_date = date            
            
            utterances = script.split('\n')
            post_utterances = []
            temp_utter = ''

            chatbot_utters = []
            characters = []
            
            for utter in utterances:
                if len(utter.strip()) == 0:
                    continue
                if 'Teleplay: ' in utter or 'Story: ' in utter:
                    continue
                if ':' in utter:
                    characters.append(utter.split(':')[0].strip())
                if chatbot+':' in utter:
                    chatbot_utters.append(utter.strip())
                if ':' in utter:
                    post_utterances.append(utter.strip())
                    temp_utter = deepcopy(utter.strip())
                else:
                    post_utterances.pop()
                    temp_utter += '\n'+utter.strip()
                    post_utterances.append(temp_utter)
            
            if sc_num != session_nums[0]:
                print()

            print('###########################################')
            print(f'Date: {date}, Conversation #{cur_conv_num}')
            print('###########################################\n')

            try:
                if len(chatbot_utters) > 1:
                    chatbot_utters = chatbot_utters[1:]
                random_chatbot_utter = random.choice(chatbot_utters)
                bot_indices = [i for i, s in enumerate(post_utterances) if random_chatbot_utter in s]
                range_indices = [i for i in range(max(0, bot_indices[0]-3), min(len(post_utterances), bot_indices[0]+3))]
                close_chars = []
                for idx in range_indices:
                    if ':' in post_utterances[idx]:
                        close_chars.append(post_utterances[idx].split(':')[0])
                characters = list(set(close_chars))
                close_chars = list(set(close_chars))
                
                for char_ in close_chars:
                    if chatbot.lower() in char_.lower() or 'all' == char_.lower():
                        try:
                            characters.remove(char_)
                        except:
                            pass 
            except:
                pass

            if len(characters) > 0:
                char_ask = random.choice(characters)
            else:
                char_ask = ""

            history_num = 0
            script_history = ""

            for un, utter_post in enumerate(post_utterances):
                print(name_change(script_name, utter_post, name_shuffle))
                history = ""
                if history_type == "utts":
                    history = name_change(script_name, utter_post, name_shuffle)
                elif history_type == "session-entire":
                    if not utter_post.endswith("\n"):
                        utter_post += "\n"
                    script_history += name_change(script_name, utter_post, name_shuffle)
                    history = script_history
                elif history_type == "session-summary":
                    if not utter_post.endswith("\n"):
                        utter_post += "\n"
                    script_history += name_change(script_name, utter_post, name_shuffle)
                    history = script_history
                else:
                    return AssertionError("Incorrect `history_type`.")

                embedding_vec = None
                
                save_timeout_flag = False
                search_timeout_flag = False
                ans_timeout_flag = False
                save_start_time = None
                save_end_time = None
                save_time = None
                
                # below are what we are actually going to log
                time_in_saving = None 
                time_in_retrieval_searching = None
                time_in_answering = None
                result_time = None
                ans_time = None
                answer = ""
                
                already_pop = False
                history_before_save_len = None
                embedding_before_save_len = None
                history_after_save_len = None
                embedding_after_save_len = None
                save_start_time = time.time()
                save_result = None
                
                try:
                    history_before_save_len = len(data_dict['history'])
                    embedding_before_save_len = len(data_dict['ada_embedding'])
                    save_result, history_num = func_timeout(sleep_time, save_history, args=(history_num, history, history_type, date, cur_conv_num, un, post_utterances, utter_post, model_name, openai_client, model, tokenizer, config, data_dict, ret_method, llama_tokenizer)) 
                    save_end_time = time.time()
                    save_time = save_end_time - save_start_time  
                    
                    
                
                except FunctionTimedOut:
                    history_after_save_len = len(data_dict['history'])
                    embedding_after_save_len = len(data_dict['ada_embedding'])
                    save_timeout_flag = True
                    print("\nTimeout (saving history)!!!\n")
                    print("Corresponding history couldn't be saved.\n")
                    if len(data_dict['history']) > 0 and history_after_save_len > history_before_save_len:
                        data_dict['history'].pop()
                    if ret_method in ["openai-emb", "no_ret"]:
                        if len(data_dict['ada_embedding']) > 0 and embedding_after_save_len > embedding_before_save_len:
                            data_dict['ada_embedding'].pop()
                        if ret_method == "openai-emb":
                            save_result = pd.DataFrame(data_dict)
                    if ret_method == "bm25":
                        if len(data_dict['history']) > 0:
                            tokenized_docs = [word_tokenize(doc.lower()) for doc in data_dict['history']]
                            save_result = BM25Okapi(tokenized_docs)
                    ret_histories = "No history.\n"
                    already_pop = True
                    result = "Wrong (Timeout in saving history)"
                    is_ambiguous = False
                    answer = "<<<Timeout in saving history>>>"
                    time_in_saving = "<<<Timeout in saving history>>>" 
                    time_in_retrieval_searching = "<<<Timeout in saving history>>>"
                    time_in_ans = "<<<Timeout in saving history>>>"
                    result_time = "<<<Timeout in saving history>>>"
                
                #### Question
                if random_chatbot_utter.lower() in utter_post.lower() and len(characters) > 0 and target_question_list[0] != 'cannot ask':
                    if already_asked == 1:
                        continue
                    real_question = ''
                    real_tar_id = -1
                    for tar_id in range(len(target_question_list)):
                        if char_ask in list(target_question_list[tar_id]['questions']):
                            real_question = target_question_list[tar_id]['questions'][char_ask]
                        elif 'default' in list(target_question_list[tar_id]['questions']):
                            real_question = target_question_list[tar_id]['questions']['default']
                        else:
                            continue
                        
                        try:
                            true_answer = target_question_list[tar_id]['answer']
                            real_tar_id = tar_id
                            assert(len(target_dates_list)==len(target_question_list))
                            gt_sessions = extract_gt_sessions_bm25_date(date_to_sessions, epi_session_date_to_sessions, current_type, target_dates_list[tar_id], epi, sc_num, num_ret_history, real_question)
                            break
                        except:
                            continue
                    
                    if real_question == '' or real_tar_id == -1 or gt_sessions == "":
                        continue
                    
                    true_answer_op = ''

                    for oi, op in enumerate(['(A)', '(B)', '(C)', '(D)', '(E)']):
                        if true_answer.lower() == target_question_list[real_tar_id]['options'][oi].lower():
                            true_answer_op = op
                            break

                    if true_answer_op == '':
                        continue
                    
                    if answer_format in ['multi_choice_unstructured', 'open_ended']:
                        if true_answer_op == "(E)":
                            true_answer_op = "I don't know."
                        else:
                            true_answer_op = true_answer
                        

                    question_part_prompt = ''

                    question_part_prompt += f'{char_ask}: {real_question}'
                    options = target_question_list[real_tar_id]['options']
                    if answer_format == 'multi_choice_structured':
                        question_part_prompt += '\n'
                        question_part_prompt += f'\t(A) {options[0]}\n'
                        question_part_prompt += f'\t(B) {options[1]}\n'
                        question_part_prompt += f'\t(C) {options[2]}\n'
                        question_part_prompt += f'\t(D) {options[3]}\n'
                        question_part_prompt += f'\t(E) {options[4]}'
                    elif answer_format == 'open_ended':
                        pass
                    elif answer_format == 'multi_choice_unstructured':
                        question_part_prompt += ' '
                        question_part_prompt += f'{options[0]}? or '
                        question_part_prompt += f'{options[1]}? or '
                        question_part_prompt += f'{options[2]}? or '
                        question_part_prompt += f'{options[3]}? or '
                        question_part_prompt += f"you don't know?"
                    else:
                        raise ValueError("Invalid answer format. Should be one of ('multi_choice_structured', 'multi_choice_unstructured', 'open_ended')")
                    question_part_prompt_sh = name_change(script_name, question_part_prompt, name_shuffle)
                    """Start of Answering. Time measure starts HERE"""
                    # time measure START
                    ans_timeout_flag = False
                    retrieve_save_start_time = None
                    ans_start_time = None
                    
                    char_ask_sh = name_change(script_name, char_ask, name_shuffle)
                    real_question_sh = name_change(script_name, real_question, name_shuffle)
                    
                    if not save_timeout_flag:
                        ret_search_start_time = time.time()
                        try:
                            ret_histories = func_timeout(sleep_time-save_time, retrieve_history, args=(ret_method, num_ret_history, openai_client, max_token_len, save_result, char_ask_sh, real_question_sh, data_dict, gt_sessions))
                            retrieve_search_time = time.time()-ret_search_start_time
                        except FunctionTimedOut: # timeout during searching history. Note that saving history was done correctly though.
                            ret_histories = "No history.\n"
                            print("\nTimeout (searching history)!!!\n")
                            search_timeout_flag = True
                            result = "Wrong (Timeout in searching history)"
                            is_ambiguous = False
                            answer = "<<<Timeout in searching history>>>"
                            time_in_saving = save_time # record actual time taken in saving
                            time_in_retrieval_searching = "<<<Timeout in searching history>>>"
                            time_in_ans = "<<<Timeout in searching history>>>"
                            result_time = "<<<Timeout in searching history>>>"
                        if not search_timeout_flag:
                        # Model inference
                            #question_part_prompt_sh = name_change(script_name, question_part_prompt, name_shuffle)
                            chatbot_sh = name_change(script_name, chatbot, name_shuffle)
                            if answer_format not in ['multi_choice_structured', 'multi_choice_unstructured', 'open_ended']:
                                raise ValueError("Invalid answer format. Should be one of ('multi_choice_structured', 'multi_choice_unstructured', 'open_ended')")
                            if ret_method == 'no_ret':
                                prompt = open_file(f'./prompt/naive_llm_inference_{answer_format}.txt').replace('<<<Date>>>', date).replace('<<<Dialog_History>>>', ret_histories).replace('<<<Question>>>', question_part_prompt_sh).replace('<<<Chatbot>>>', chatbot_sh)
                            else:
                                prompt = open_file(f'./prompt/RAG_qa_prompt_{answer_format}.txt').replace('<<<Date>>>', date).replace('<<<Dialog_History>>>', ret_histories).replace('<<<Question>>>', question_part_prompt_sh).replace('<<<Chatbot>>>', chatbot_sh)
                        
                            ans_start_time = time.time()
                            try:
                                answer = func_timeout(sleep_time-save_time-retrieve_search_time, answer_question, args=(model_name, client, model, tokenizer, config, prompt))
                                ans_time = time.time() - ans_start_time
                                time_in_saving = save_time       
                                time_in_retrieval_searching = retrieve_search_time
                                time_in_answering = ans_time
                                result_time = save_time + retrieve_search_time + ans_time
                            except FunctionTimedOut:
                                print("\nTimeout (answering)!!!\n")
                                ans_timeout_flag = True
                                result = "Wrong (Timeout in answering)"
                                is_ambiguous = False
                                answer = "<<<Timeout in answering>>>"
                                time_in_saving = save_time
                                time_in_retrieval_searching = retrieve_search_time
                                time_in_answering = "<<<Timeout in answering>>>"
                                result_time = "<<<Timeout in answering>>>"
                            """Measuring time for timeout stops HERE"""
                    
                    is_ambiguous = False
                    if not ans_timeout_flag and not save_timeout_flag and not search_timeout_flag:
                        result, is_ambiguous = judge_eq(true_answer_op, answer, question_part_prompt_sh, client, answer_format=answer_format)
                        if result_time >= sleep_time:
                            result = "Wrong (Timeout)"
                        else:
                            if fast_eval == "no":
                                time.sleep(sleep_time-result_time)

                    already_asked = 1
                    # log results
                    answer_list.append(answer)
                    gold_answer_list.append(true_answer_op)
                    result_list.append(result)
                    result_time_list.append(result_time)
                    save_time_list.append(time_in_saving)
                    retrieve_search_time_list.append(time_in_retrieval_searching)
                    ans_time_list.append(time_in_answering)
                    target_level_list.append({"current_type" : current_type})
                    print(question_part_prompt_sh)
                    print(f'------------------------------- Q&A result -------------------------------')
                    print(f'result: {result}, ambiguous answer: {is_ambiguous}')
                    print(f'true answer: {true_answer_op}\t model answer: {answer}')
                    print(f'time spent in saving: {time_in_saving}')
                    print(f'time spent in searching history: {time_in_retrieval_searching}')
                    print(f'time spent in answering: {time_in_answering}')
                    print(f'time spent overall: {result_time}')
                    print(f'time limit: {sleep_time}')
                    print(f'model name: {model_name}')
                    print(f'--------------------------------------------------------------------------')
                    
                    if is_ambiguous:
                        ambiguous_idx_list.append((epi, sc_num, question_part_prompt_sh))
                        ambiguous_answer_list.append(answer)
                        ambiguous_gold_answer_list.append(true_answer_op)

                    distilled_answer = distill_answer(answer)
                    ret_histories_question_answer_list.append((ret_histories, question_part_prompt_sh, true_answer_op, distilled_answer))
                    
                    calibration = calibrate(result, is_ambiguous, true_answer_op, answer, question_part_prompt_sh, distilled_answer, answer_format=answer_format, lenient=True) # (result, is_ambiguous, calibrated_distilled_answer)
                    if isinstance(result_time, float) and result_time >= sleep_time:
                        calibrated_result_list.append("Wrong (Timeout)")
                        calibrated_distilled_answer_list.append("Wrong (Timeout)")
                    else:
                        calibrated_result_list.append(calibration[0])
                        calibrated_distilled_answer_list.append(calibration[2])
                    
                else:
                    if fast_eval == "no":
                        if save_time == None:
                            pass
                        else:
                            time.sleep(sleep_time-save_time)
                
                if not already_pop and "session" in history_type and un < len(post_utterances) - 1:
                    if ret_method == 'openai-emb' or ret_method == 'no_ret':
                        try:
                            data_dict["history"].pop()
                            data_dict["ada_embedding"].pop()
                        except:
                            AssertionError("Unexpected error(probable cause: couldn't save even one embedding using openai-emb in time). Please run the program again.")
                    else:
                        try:
                            data_dict["history"].pop()
                        except:
                            pass
            cur_conv_num += 1
    
    simulator_running_time = time.time() - simulator_start_time
    
    if "Correct" in result_list:
        score_total = result_list.count('Correct') / len(result_list)
    else:
        score_total = 0
    
    valid_result_time_list = []
    for result_time in result_time_list:
        if isinstance(result_time, float):
            valid_result_time_list.append(result_time)
    
    if len(valid_result_time_list) == 0:
        result_time_mean = 0
    else:
        result_time_mean = sum(valid_result_time_list) / len(valid_result_time_list)
    
    if "Correct" in calibrated_result_list:
        calibrated_score = calibrated_result_list.count('Correct') / len(calibrated_result_list)
    else:
        calibrated_score = 0
    
    log_info = {
        "score" : score_total,
        "calibrated_score" : calibrated_score,
        "result_time_mean" : result_time_mean,
        "simulator_running_time" : simulator_running_time,
        "result_list" : result_list,
        "result_time_list" : result_time_list,
        "ambiguous_idx_list" : ambiguous_idx_list,
        "ambiguous_answer_list" : ambiguous_answer_list,
        "ambiguous_gold_answer_list" : ambiguous_gold_answer_list,
        "answer_list" : answer_list,
        "gold_answer_list" : gold_answer_list,
        "ret_histories_question_answer_list" : ret_histories_question_answer_list,
        "save_time_list" : save_time_list,
        "retrieve_search_time_list": retrieve_search_time_list, 
        "ans_time_list" : ans_time_list,
        "calibrated_result_list" : calibrated_result_list,
        "calibrated_distilled_answer_list" : calibrated_distilled_answer_list,
        "target_level_list" : target_level_list
    }
        
    return log_info



if __name__ == "__main__":
    args = parse_args()
    print(args)
    def set_affinity(num_cores, sh_number):
        cpu_list = range(num_cores*sh_number, num_cores*(sh_number+1))
        os.sched_setaffinity(os.getpid(), set(cpu_list))
        
    set_affinity(args.num_cores, args.sh_number)
    cpu_count = os.sched_getaffinity(os.getpid())
    print(f"Available CPUs: {cpu_count}")
    
    log_info = simulator(script_name=args.script_name, history_type=args.history_type, sleep_time=args.sleep_time, num_ret_history=args.num_ret_history, model_name=args.model_name, \
                        debug=args.debug, debug_n_episodes=args.debug_n_episodes, quantization=args.quantization, ret_method=args.ret_method, name_shuffle=args.name_shuffle, openai_api_key=args.openai_api_key, gemini_api_key=args.gemini_api_key, antrhopic_api_key=args.antrhopic_api_key, fast_eval=args.fast_eval, answer_format=args.answer_format)

    print()
    print('SCORE: ', log_info["score"])
    print(f'SCORE(calibrated): {log_info["calibrated_score"]}')
    print('Answer Time Mean: ', log_info["result_time_mean"])
    
    log_results_path = \
        f"./results/results-{args.script_name}-model_{args.model_name}-debug_{args.debug}-quantization_{args.quantization}-time_limit_{args.sleep_time}-history_type_{args.history_type}-{args.ret_method}_{args.name_shuffle}-version_{args.trial_version}.json"
    log_total_path = \
        f"./results/entire_log-{args.script_name}-model_{args.model_name}-debug_{args.debug}-quantization_{args.quantization}-time_limit_{args.sleep_time}-history_type_{args.history_type}-{args.ret_method}_{args.name_shuffle}-version_{args.trial_version}.json"
    
    log_results(log_info, log_file_path=log_results_path)
    log_everything(log_info, log_file_path=log_total_path)