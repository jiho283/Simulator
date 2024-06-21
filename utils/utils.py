from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

    
def extract_gt_sessions_bm25_date(date_to_sessions, epi_scene_date_to_sessions, current_type, target_dates, epi, sc_num, num_history, question):
    gt_sessions = ""
    cur_conv_num = 0
    total_history = 0
    gt_docs = []
    if current_type in ['ans_w_time', 'dont_know_unans_time']:
        if len(target_dates) == 0:
            gt_sessions = "No Relevant History\n"
            return gt_sessions
        sessions = date_to_sessions[target_dates[0]]
        for session in sessions:
            cur_conv_num += 1
            gt_sessions += f'[Date: {target_dates[0]}, Session #{cur_conv_num}]{session}'
            total_history += 1
            gt_docs.append(f'[Date: {target_dates[0]}, Session #{cur_conv_num}]{session}')
            #if total_history == num_history:
            #    return gt_sessions
	
    elif current_type in ['ans_wo_time', 'before_event_unans', 'dont_know_unans']:
        #import pdb; pdb.set_trace()
        scene_nums = list(epi_scene_date_to_sessions[epi])
        before_date = ''
        for session_num in scene_nums:
            date = list(epi_scene_date_to_sessions[epi][session_num])[0]
            if date != before_date:
                cur_conv_num = 0
            before_date = date
            session = epi_scene_date_to_sessions[epi][session_num][date]
            cur_conv_num += 1
            
            gt_sessions += f'[Date: {date}, Session #{cur_conv_num}]{session}'
            total_history += 1
            gt_docs.append(f'[Date: {date}, Session #{cur_conv_num}]{session}')
            #if total_history == num_history:
            #    return gt_sessions
    else:
        for date in target_dates:
            if 'fu' in date:
                continue
            sessions = date_to_sessions[date]
            cur_conv_num = 0
            for session in sessions:
                cur_conv_num += 1
                gt_sessions += f'[Date: {date}, Session #{cur_conv_num}]{session}'
                total_history += 1
                gt_docs.append(f'[Date: {date}, Session #{cur_conv_num}]{session}')
                #if total_history == num_history:
                #    return gt_sessions
    if gt_sessions == "":
        gt_sessions = "No Relevant History\n"
        return gt_sessions
    tokenized_docs = [word_tokenize(gt_instance.lower()) for gt_instance in gt_docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_question = word_tokenize(question.lower())
    doc_scores = bm25.get_scores(tokenized_question)
    top_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:num_history]
    top_docs = [gt_docs[i] for i in top_doc_indices]
    top_docs_date = sorted(top_docs, key=lambda x: x.split("]")[0], reverse=False)
    result = "".join(top_docs_date)
    return result


def get_embedding(text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_history(df, product_description, client, n=5, model="text-embedding-3-small"):
    embedding = get_embedding(product_description, client, model=model)
    embeddings_matrix = np.vstack(np.array(df.ada_embedding.values))
    similarities = cosine_similarity(embeddings_matrix, np.array(embedding)[None, :])
    df['similarities'] = similarities.flatten()
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def name_change(script, input, mode):
    
    shuffle_mapping_dict = {
        'friends': {'Monica': 'Joey',
                    'Chandler': 'Rachel',
                    'Joey': 'Monica',
                    'Phoebe': 'Ross',
                    'Rachel': 'Chandler',
                    'Ross': 'Phoebe'},
        'bigbang': {'Howard': 'Amy',
                    'Leonard': 'Howard',
                    'Raj': 'Penny',
                    'Penny': 'Raj',
                    'Bernadette': 'Sheldon',
                    'Amy': 'Leonard',
                    'Sheldon': 'Bernadette'},
        'theoffice': {'Dwight': 'Ryan',
                    'Jim': 'Michael',
                    'Pam': 'Dwight',
                    'Ryan': 'Jim',
                    'Michael': 'Pam'}
    }

    new_name_mapping_dict = {
        'friends': {'Monica': 'Patricia',
                    'Chandler': 'James',
                    'Joey': 'John',
                    'Phoebe': 'Jennifer',
                    'Rachel': 'Linda',
                    'Ross': 'Robert'},
        'bigbang': {'Howard': 'Robert',
                    'Leonard': 'James',
                    'Raj': 'Michael',
                    'Penny': 'Jennifer',
                    'Bernadette': 'Linda',
                    'Amy': 'Patricia',
                    'Sheldon': 'John'},
        'theoffice': {'Dwight': 'John',
                    'Jim': 'Robert',
                    'Pam': 'Jennifer',
                    'Ryan': 'William',
                    'Michael': 'James'}
    }
    
    if mode == 'shuffle':
        chars = list(shuffle_mapping_dict[script])
        sub_dict = {}
        for i, char in enumerate(chars):
            sub_dict[f'[MASK {i}]'] = shuffle_mapping_dict[script][char]
            input = input.replace(char, f'[MASK {i}]')
        for i, char in enumerate(chars):
            input = input.replace(f'[MASK {i}]', sub_dict[f'[MASK {i}]'])
    
    elif mode == 'new_name':
        chars = list(new_name_mapping_dict[script])
        sub_dict = {}
        for i, char in enumerate(chars):
            sub_dict[f'[MASK {i}]'] = new_name_mapping_dict[script][char]
            input = input.replace(char, f'[MASK {i}]')
        for i, char in enumerate(chars):
            input = input.replace(f'[MASK {i}]', sub_dict[f'[MASK {i}]'])
    
    return input


    
