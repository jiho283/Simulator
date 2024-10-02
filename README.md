# DialSim

We introduce DialSim, a real-time dialogue simulator. In this simulator, an agent is assigned the role of a character from popular TV shows, requiring it to respond to spontaneous questions using past dialogue information and to distinguish between known and unknown information. Key features of DialSim include evaluating the agent’s ability to respond within a reasonable time limit, handling long-term multi-party dialogues, and managing adversarial settings (e.g., swap character names) to challenge the agent’s reliance on pre-trained knowledge. 

## Dataset
You can download the dataset [here](https://drive.google.com/drive/folders/1MhPlUFWuchVZ5E1NQDWfbT7_RW7ozbuk?usp=drive_link).

v1.0: This version includes the dataset as described in the paper.

v1.1: To incorporate more diverse and challenging data, this version has been updated to include unanswerable multi-hop questions.

## Experimental Setup

After downloading appropriate version of ```torch```, do:

1. ```pip install -r requirements.txt```

2. ```mkdir data```

3. ```mv dialsim_v1.1.zip ./data/```

4. ```cd data```

5. ```unzip dialsim_v1.1.zip```

## Simulation
Command Example:
```CUDA_VISIBLE_DEVICES=0 python simulator.py --model_name "gpt-3.5-turbo" --quantization "4bit" --script_name "friends" --sleep_time 6 --history_type "session-entire" --ret_method "bm25"  --trial_version 0 --sh_number 0 --num_cores 10 --answer_format "multi_choice_structured" --openai_api_key  "<<YOUR_OPENAI_API_KEY>>(not required in this line)"```

```CUDA_VISIBLE_DEVICES=0 python simulator.py --model_name "gpt-4o" --quantization "4bit" --script_name "friends" --sleep_time 6 --history_type "session-summary" --ret_method "bm25"  --trial_version 0 --sh_number 0 --num_cores 10 --answer_format "open_ended" --openai_api_key  "<<YOUR_OPENAI_API_KEY>>(required in this line)"```

```CUDA_VISIBLE_DEVICES=0 python simulator.py --model_name "gpt-4o-mini" --quantization "4bit" --script_name "friends" --sleep_time 6 --history_type "utts" --ret_method "openai-emb"  --trial_version 0 --sh_number 0 --num_cores 10 --answer_format "multi_choice_unstructured" --openai_api_key  "<<YOUR_OPENAI_API_KEY>>(required in this line)"```

```CUDA_VISIBLE_DEVICES=0 python simulator.py --model_name "llama2-7b-chat" --quantization "4bit" --script_name "friends" --sleep_time 6 --ret_method "no_ret"  --trial_version 0 --sh_number 0 --num_cores 10 --answer_format "multi_choice_unstructured" --openai_api_key  "<<YOUR_OPENAI_API_KEY>>(required in this line)"```

#### Arguments
- `model_name`: Specifies the model to use, default is "gpt-3.5-turbo". Options include "llama2-7b-chat", "llama2-70b-chat", "tulu2-7b-dpo", "tulu2-70b-dpo", "gemma-2b-it", "gemma-7b-it", "mistral-7b-it", "mixtral-it", "claude-3", "claude-2.1", and model names for openai models and gemini models.
- `quantization`: Model quantization level, default is "no". Options include "no", "16bit", "8bit", and "4bit".
- `script_name`: TV show script for the simulation, default is "friends". Options include "friends", "bigbang", and "theoffice".
- `sleep_time`: Response time limit, default: 5
- `history_type`: Method for saving history, default is "session-entire". Options include "utts", "session-entire", and "session-summary".
- `num_ret_history`: Number of retrieved histories to use. Modify lines 180-198, 222-236 in `simulator.py` to change this number.
- `ret_method`: Retrieval method, default is "bm25". Options include "openai-emb", "bm25", "no_ret", and "oracle".
- `name_shuffle`: Type of adversarial test, default is "original". Options include "original", "shuffle", and "new_name".
- `answer_format`: The format of the answer the model to generate, default is "multi_choice_structured". Options include "multi_choice_structured", "multi_choice_unstructured", "open_ended".
- `trial_version`: Experiment version number, default: 0
- `sh_number`: Shell script number, default: 0
- `num_cores`: Maximum number of CPU cores to use, default: 10
- `openai_api_key`: Required if using openai models, or `ret_method="openai-emb"`, or using "multi_choice_unstructured" or "open_ended" for `answer_format`(we use gpt-4o-mini as a judge when `answer_format` is "multi_choice_unstructured" or "open_ended", and rule-based evaluation when `answer_format` is "multi_choice_structured").
- `gemini_api_key`: Required if using "gemini" in the model name.
- `anthropic_api_key`: Required if using "claude-3" or "claude-2.1" in the model name.
- `fast_eval`: When set to "yes", the simulator proceeds to the next utterance without waiting for the time interval if the history has already been updated. The default setting is "yes". Options include "yes" and "no".
    debug=True,
    debug_n_episodes=2,
    fast_eval=True
)

custom_simulator.simulate()
log_info = custom_simulator.log_results()
custom_simulator.save_log(log_info)
```
