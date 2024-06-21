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
```CUDA_VISIBLE_DEVICES=0 python simulator.py --model_name "GPT-3.5" --quantization "4bit" --script_name "friends" --ret_method "bm25" --history_type "session-entire" --sleep_time 6 --openai_api_key "<<YOUR_OPENAI_API_KEY>>"  --trial_version 0 --sh_number 0 --num_cores 10```

#### Arguments
a) model_name: name of the model (one of "llama2-7b-chat", "llama2-70b-chat", "tulu2-7b-dpo", "tulu2-70b-dpo", "gemma-2b-it", "gemma-7b-it", "mistral-7b-it", "mixtral-it", "GPT-3.5", "GPT-4", "claude-3", "claude-2.1", and "gemini"). Default: "GPT-3.5"

b) quantization: 

c) script_name:

d) sleep_time: 

e) history_type: 

f) num_ret_history: 

g) ret_method: 

h) name_shuffle: 

i) trial_version:

j) sh_number:

k) num_cores: 

l) openai_api_key: 

m) gemini_api_key:

n) antrhopic_api_key: 

