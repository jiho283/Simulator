CUDA_VISIBLE_DEVICES=0 python simulator_func_timeout_dev.py --model_name "llama2-7b-chat" --quantization "4bit" --script_name "friends" --ret_method "bm25" --history_type "session-entire" --sleep_time 600 --trial_version 1 --sh_number 0
CUDA_VISIBLE_DEVICES=0 python simulator_func_timeout_dev.py --model_name "mixtral-it" --quantization "4bit" --script_name "friends" --ret_method "bm25" --history_type "session-entire" --sleep_time 600 --trial_version 2 --sh_number 0