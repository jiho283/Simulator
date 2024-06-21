import json
def log_results(log_info, log_file_path=""):
    if log_file_path == "":
        raise AssertionError("Need to specify log directory.")
    info = {
        "score" : log_info["score"],
        "calibrated_score" : log_info["calibrated_score"],
        "calibrated_result_list" : log_info["calibrated_result_list"],
        "avg_answer_time" : log_info["result_time_mean"],
        "result_list" : log_info["result_list"],
        "result_time_list" : log_info["result_time_list"],
        "ambiguous_idx_list" : log_info["ambiguous_idx_list"],
        "ambiguous_answer_list" : log_info["ambiguous_answer_list"],
        "ambiguous_gold_answer_list" : log_info["ambiguous_gold_answer_list"]
    }
    with open(log_file_path, "w") as f:
        json.dump(info, f, indent=2)

def log_answers(answer_list, gold_answer_list, log_file_path=""):
    if log_file_path == "":
        raise AssertionError("Need to specify log directory.")
    info = {
        "answer_list" : answer_list,
        "gold_answer_list" : gold_answer_list
    }
    with open(log_file_path, "w") as f:
        json.dump(info, f, indent=2)

def log_ret_history_question(ret_history_question_answer_list, target_level_list, log_file_path=""):
    if log_file_path == "":
        raise AssertionError("Need to specify log directory.")
    info = []
    for (ret_history, question, gold_answer, distilled_answer) in ret_history_question_answer_list:
        info.append({
            "ret_history" : ret_history,
            "question" : question,
            "gold_answer" : gold_answer,
            "model_answer" : distilled_answer,
            "target_level_list" : target_level_list
        }) 
    with open(log_file_path, "w") as f:
        json.dump(info, f, indent=2)

def log_times(save_time_list, retrieve_search_time_list, ans_time_list,  log_file_path=""):
    if log_file_path == "":
        raise AssertionError("Need to specify log directory.")
    info = []
    for (save_time, retrieve_search_time, ans_time) in zip(save_time_list,retrieve_search_time_list, ans_time_list):
        info.append({
            "save_time" : save_time,
            "retrieve_search_time" : retrieve_search_time,
            "ans_time" : ans_time
        }) 
    with open(log_file_path, "w") as f:
        json.dump(info, f, indent=2)

def log_calibration(log_info, log_file_path=""):
    if log_file_path == "":
        raise AssertionError("Need to specify log directory.")
    info = []
    for (calibrated_result, calibrated_distilled_answer) in zip(log_info["calibrated_result_list"], log_info["calibrated_distilled_answer_list"]):
        info.append({
            "calibrated_result" : calibrated_result,
            "calibrated_distilled_answer" : calibrated_distilled_answer
        }) 
    with open(log_file_path, "w") as f:
        json.dump(info, f, indent=2)
        
def log_everything(log_info, log_file_path=""):
    if log_file_path == "":
        raise AssertionError("Need to specify log directory.")
    with open(log_file_path, "w") as f:
        json.dump(log_info, f, indent=2)