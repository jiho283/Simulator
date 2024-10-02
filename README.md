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


## Python Package
You can also install the package with
```pip install dialsim```

### How to Use
```
# import necessary modules
from dialsim.models.load_model import load_model
from dialsim.agent import DialSim
from dialsim.agent import Agent

# specify the name of the script ("friends", "theoffice", "bigbang")
script_name = "friends"
# specify model name
model_name = "gpt-4o-mini"
client = OpenAI(api_key="<<Your OpenAI API KEY>>")

model, tokenizer, config = load_model(model_name, "4bit")


# create agent
agent = Agent(
    history_type="session-entire",
    ret_method="bm25",
    num_ret_history=20,
    model_name=model_name, 
    model=model,
    tokenizer=tokenizer, # None if using API based models
    config=config, # None if using API based models
    client=client,
    openai_client=client,
    answer_format="multi_choice_structured" # "multi_choice_structured", "multi_choice_unstructured", "open_ended"
)

# create simulator
simulator = DialSim(
    sleep_time=50,
    script_name=script_name,
    agent=agent,
    name_shuffle="original", # "original", "new_name", "shuffle"
    tkg_ratio=0.7,
    debug=True, # debug mode: you can limit the number of episodes
    debug_n_episodes=2,
    fast_eval=True # if you don't want to wait for sleep_time seconds for each utterance of the speakers
)

# run simulation
simulator.simulate()
# get log
log_info = simulator.log_results()
# save log. Default path is "./results/log.json"
simulator.save_log(log_info)
```

### Customized Agents
You can also use your own customized agents by overriding the methods in `Agent` and `Dialsim`:
```
# example: customized way of saving history. Rest is the same as the baselines in the paper.
from dialsim.models.api_based_inference import gpt_inference
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from dialsim.utils import get_embedding, search_history, open_file, name_change
class CustomAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def save_history(self, generator_instance) -> tuple:
        #return super().save_history(history, date, cur_conv_num, un, post_utterances)
        if self.history_type == "openie":
            un = generator_instance["un"]
            post_utterances = generator_instance["post_utterances"]
            prev_cnt = min(un, 3)
            prev_utts = post_utterances[un-prev_cnt : un]
            history = '\n'.join(prev_utts)
            generator_instance["history"] = history
            openie_prompt = open_file('./data/openie_utt_prompt.txt').replace('<<<PREV_UTTS>>>', generator_instance["history"]).replace('<<<LAST_UTT>>>', generator_instance["utter_post_sh"])
            cur_triples = gpt_inference(message=openie_prompt, model_name=self.model_name, client=self.openai_client).replace(";","")
            processed_history = f'[Date: {generator_instance["date"]}, Session #{generator_instance["cur_conv_num"]}, History #{generator_instance["history_num"]+1}]\n{cur_triples}'
            generator_instance["history"] = processed_history
            
            self.data_dict['history'].append(processed_history)
            self.is_data_dict_history_updated = True
            if self.ret_method == 'openai-emb':
                embedding_vec = get_embedding(processed_history, client=self.client, model="text-embedding-3-small")
                self.data_dict['ada_embedding'].append(embedding_vec)
                self.is_data_dict_embedding_updated = True
                data_df = pd.DataFrame(self.data_dict)
                return data_df
            elif self.ret_method == 'bm25':
                tokenized_docs = [word_tokenize(doc.lower()) for doc in self.data_dict['history']]
                bm25 = BM25Okapi(tokenized_docs)
                return bm25
            elif self.ret_method == 'no_ret':
                token_len = self.llama_tokenizer(processed_history, return_tensors="pt", truncation=True).input_ids.shape[1]
                self.data_dict['ada_embedding'].append(token_len)
                self.is_data_dict_embedding_updated = True
                return None
            elif self.ret_method == "oracle":
                return None
            else:
                raise ValueError("Incorrect `ret_method`.")
        else:
            raise NotImplementedError("Only `openie` history type is supported.")

# customized simulator that processes each instance of the simulation.
class CustomSimulator(DialSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def process_walk(self):
        un = self.generator_instance["un"]
        post_utterances = self.generator_instance["post_utterances"]
        prev_cnt = min(un, 3)
        prev_utts = post_utterances[un-prev_cnt : un]
        history_before = '\n'.join(prev_utts)
        history = name_change(self.script_name, history_before, self.name_shuffle)
        self.generator_instance["history"] = history
        return self.generator_instance

# the rest is the same
custom_agent = CustomAgent(
    history_type="openie",
    ret_method="openai-emb",
    num_ret_history=20,
    model_name=model_name, 
    model=model,
    tokenizer=tokenizer,
    config=config,
    client=client,
    openai_client=client,
    adjust_num_ret_history_=False,
    answer_format="multi_choice_structured"
)

custom_simulator = CustomSimulator(
    sleep_time=5,
    script_name=script_name,
    agent=custom_agent,
    name_shuffle="original",
    tkg_ratio=0.7,
    debug=True,
    debug_n_episodes=2,
    fast_eval=True
)

custom_simulator.simulate()
log_info = custom_simulator.log_results()
custom_simulator.save_log(log_info)
```
