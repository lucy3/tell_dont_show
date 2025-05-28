import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers, os, sys, json
import torch
from collections import defaultdict
from datasets import Dataset
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from openai import OpenAI 
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

#LOGS = '/data/lucy/literary-theme-analysis/outputs/' # tahoe
LOGS = '/mnt/data0/lucy/literary-theme-analysis/outputs/' # sequoia

def get_data(all_messages):
    for i in range(len(all_messages)): 
        yield all_messages[i]

def run_abstraction_hf(abstraction_type, model_name):
    '''
    Temperature and top_p are defaults in llama 3 github repo
    '''
    with open(os.path.join(LOGS, 'book_passages_augmented.json'), 'r') as infile: 
        book_passages = json.load(infile) # {book : {quote_id : text} }
    
    if model_name == 'Llama-3.1-70B': 
        model_id = "/data/models/llama3.1/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693"
        model_kwargs = {"torch_dtype": torch.bfloat16}
        torch_dtype = torch.bfloat16
    elif model_name == 'Llama-3.1-8B': 
        model_id = "/data/models/llama3.1/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
        model_kwargs = {'torch_dtype': torch.bfloat16}
    elif model_name == 'Phi-3.5-mini': 
        model_id = "/data/models/phi/models--microsoft--Phi-3.5-mini-instruct/snapshots/af0dfb8029e8a74545d0736d30cb6b58d2f0f3f0"
        model_kwargs = {'torch_dtype': "auto", 'attn_implementation': 'flash_attention_2'}
    elif model_name == 'gemma-2-2b': 
        model_id = "/data/models/gemma/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
        model_kwargs = {'torch_dtype': torch.bfloat16}
    else: 
        return
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs=model_kwargs,
        device_map="auto",
    )

    #terminators = [
    #    pipeline.tokenizer.eos_token_id,
    #    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    #]
    
    outpath = os.path.join(LOGS, 'abstracted_passages_' + abstraction_type + '_' + model_name + '.json')
    
    if not os.path.isfile(outpath):
        i = 0
        abstracted_passages = {}
    else: 
        with open(outpath, 'r') as infile: 
            abstracted_passages = json.load(infile)

    print("Running...", model_name)
    for j, book in enumerate(book_passages):
        if book in abstracted_passages: 
            continue
        print("Book", j, "out of", len(book_passages))
        abstracted_passages[book] = {}
            
        all_messages = []
        passage_order = sorted(book_passages[book].keys())
        print("Number of passages:", len(passage_order))
        for passage_id in passage_order: 
            passage = book_passages[book][passage_id]
            passage_len = len(passage.split())
            if passage_len > 5000: # overly long (e.g. strange bible passages)
                passage = ' '.join(passage.split(' ')[:5000])
            prompt="In one paragraph, " + abstraction_type + " the following book excerpt for a literary scholar analyzing narrative content. Do not include the book title or author’s name in your response; " + abstraction_type + " only the passage.\n\nPassage:\n" + passage + "\n"
            messages = [
                {"role": "system", "content": "You are a helpful assistant; follow the instructions in the prompt."},
                {"role": "user", "content": prompt},
            ]   
            if model_name.startswith('gemma'): # system role not supported
                messages = [messages[1]]
            all_messages.append(messages)
        
        if len(all_messages) == 0: continue

        print("Making dataset...")
        dataset = Dataset.from_dict({"chat": all_messages})
            
        print("Running model...")
        i = 0
        if model_name.startswith('Llama'): 
            generation_args = {
                "max_new_tokens": 1024,
                "num_return_sequences": 2,
            }
        if model_name.startswith('Phi'): 
            generation_args = {
                "max_new_tokens": 1024,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
        if model_name.startswith('gemma'): 
            generation_args = {
                "max_new_tokens": 1024,
                "return_full_text": False,
            }
        for out in tqdm(pipeline(KeyDataset(dataset, "chat"),
                            **generation_args)): 
            passage_id = passage_order[i]
            assert passage_id not in abstracted_passages[book]

            abstracted_passages[book][passage_id] = out
            i += 1
            
        with open(outpath, 'w') as outfile: 
            json.dump(abstracted_passages, outfile)
            
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_with_backoff(client, model_name, messages, n):
    return client.chat.completions.create(model=model_name, messages=messages, n=n)
            
def run_abstraction_oa(abstraction_type, model_name):
    with open(os.path.join(LOGS, 'book_passages_augmented.json'), 'r') as infile: 
        book_passages = json.load(infile) # {book : {quote_id : text} }
        
    outpath = os.path.join(LOGS, 'abstracted_passages_' + abstraction_type + '_' + model_name + '.json')
    
    if not os.path.isfile(outpath):
        i = 0
        abstracted_passages = {}
    else: 
        with open(outpath, 'r') as infile: 
            abstracted_passages = json.load(infile)
        
    client = OpenAI()
    
    print("Running...")
    for j, book in enumerate(book_passages):
        if book in abstracted_passages: 
            continue
        print("Book", j, "out of", len(book_passages))
        abstracted_passages[book] = {}
            
        all_messages = []
        passage_order = sorted(book_passages[book].keys())
        print("Number of passages:", len(passage_order))
        for passage_id in passage_order: 
            passage = book_passages[book][passage_id]
            passage_len = len(passage.split())
            if passage_len > 5000: # overly long (e.g. strange bible passages)
                passage = ' '.join(passage.split(' ')[:5000])
            prompt="In one paragraph, " + abstraction_type + " the following book excerpt for a literary scholar analyzing narrative content. Do not include the book title or author’s name in your response; " + abstraction_type + " only the passage.\n\nPassage:\n" + passage + "\n"
            messages = [
                {"role": "system", "content": "You are a helpful assistant; follow the instructions in the prompt."},
                {"role": "user", "content": prompt},
            ]   
            
            completion = completion_with_backoff(client, model_name, messages, 2)
        
            abstracted_passages[book][passage_id] = completion.to_dict()
            
        with open(outpath, 'w') as outfile: 
            json.dump(abstracted_passages, outfile)
    
if __name__ == '__main__':
    run_abstraction_oa('describe', 'gpt-4o')
    
    #run_abstraction_hf('summarize', 'gemma-2-2b')
    #run_abstraction_hf('describe', 'gemma-2-2b')
    #run_abstraction_hf('paraphrase', 'gemma-2-2b')
    
    #run_abstraction_oa('describe', 'gpt-4o-mini')
    #run_abstraction_oa('paraphrase', 'gpt-4o-mini')
    #run_abstraction_hf('summarize', 'Llama-3.1-8B')
    #run_abstraction_hf('describe', 'Llama-3.1-8B')
    #run_abstraction_hf('paraphrase', 'Llama-3.1-8B')
    #run_abstraction_hf('summarize', 'Phi-3.5-mini')
    #run_abstraction_hf('describe', 'Phi-3.5-mini')
    #run_abstraction_hf('paraphrase', 'Phi-3.5-mini')
