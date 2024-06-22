# DialSim

We introduce DialSim, a real-time dialogue simulator. In this simulator, an agent is assigned the role of a character from popular TV shows, requiring it to respond to spontaneous questions using past dialogue information and to distinguish between known and unknown information. Key features of DialSim include evaluating the agent’s ability to respond within a reasonable time limit, handling long-term multi-party dialogues, and managing adversarial settings (e.g., swap character names) to challenge the agent’s reliance on pre-trained knowledge. 

The dataset is released along with our [paper](https://arxiv.org/abs/2406.13144). For further details, please refer to our paper.

## Dataset
You can download the dataset [here](https://drive.google.com/drive/folders/1MhPlUFWuchVZ5E1NQDWfbT7_RW7ozbuk?usp=sharing).

v1.0 (April 2024): This version includes the dataset as described in the paper.

v1.1 (June 2024): To incorporate more diverse and challenging data, this version has been updated to include unanswerable multi-hop questions.

## Experimental Setup

1. ```pip install -r requirements.txt```

2. ```mkdir data```

3. ```mv dialsim_v1.1.zip ./data/```

4. ```cd data```

5. ```unzip dialsim_v1.1.zip```

## Simulation
Command Example:
```CUDA_VISIBLE_DEVICES=0 python simulator.py --model_name "GPT-3.5" --quantization "4bit" --script_name "friends" --sleep_time 6 --history_type "session-entire" --ret_method "bm25"  --trial_version 0 --sh_number 0 --num_cores 10 --openai_api_key "<<YOUR_OPENAI_API_KEY>>"```

#### Arguments
- `model_name`: Specifies the model to use, default is "GPT-3.5". Options include "llama2-7b-chat", "llama2-70b-chat", "tulu2-7b-dpo", "tulu2-70b-dpo", "gemma-2b-it", "gemma-7b-it", "mistral-7b-it", "mixtral-it", "GPT-3.5", "GPT-4", "claude-3", "claude-2.1", and "gemini".
- `quantization`: Model quantization level, default is "no". Options include "no", "16bit", "8bit", and "4bit".
- `script_name`: TV show script for the simulation, default is "friends". Options include "friends", "bigbang", and "theoffice".
- `sleep_time`: Response time limit, default: 5
- `history_type`: Method for saving history, default is "session-entire". Options include "utts", "session-entire", and "session-summary".
- `num_ret_history`: Number of retrieved histories to use. Modify lines 182-240 in `simulator.py` to change this number.
- `ret_method`: Retrieval method, default is "bm25". Options include "openai-emb", "bm25", "no_ret", and "oracle".
- `name_shuffle`: Type of adversarial test, default is "original". Options include "original", "shuffle", and "new_name".
- `trial_version`: Experiment version number, default: 0
- `sh_number`: Shell script number, default: 0
- `num_cores`: Maximum number of CPU cores to use, default: 10
- `openai_api_key`: Required if using "GPT-3.5", "GPT-4" or `ret_method="openai-emb"`.
- `gemini_api_key`: Required if using "gemini" in the model name.
- `anthropic_api_key`: Required if using "claude-3" or "claude-2.1" in the model name.
