import os
import csv
from tqdm import tqdm
from blingfire import text_to_words
import json
from collections import Counter
import math

def transform_books_into_inputs(book_passages, passage_type, book_entities, out_dir): 
    '''
    book_passages is a dictionary of the form {book : {passage_id : passage_text}}

    passage_type is something like "summarize_Llama-3.1-8B_0", or a unique identifier of the type of experiment you're running

    book_entities is a dictionary of the form {book : {passage_id : [entity1, entity2, ...]}}, where each 
    entity is the name of a character detected using spacy NER. we filter these out so that our "topics" are generalizable across books
    '''  
    doc_num = 0
    out = []
    doc_num_passage_id = {} # {doc_num : book_passage_id } 
    word_doc_count = Counter() # { word : # of docs it's in } 
    
    for book in tqdm(book_passages):
        for passage_id in book_passages[book]:
            ents_tokens = set(book_entities[book][passage_id])
            passage = book_passages[book][passage_id]
                        
            doc_num_passage_id[doc_num] = book + '_' + str(passage_id) # it's useful to store some mapping from document number in the input tsv to the ID of the passage
            
            words = text_to_words(passage).split(' ') # blingfire package's tokenization function
            filtered_words = [w.lower().replace('\"', '').replace('--', ' ') for w in words if w not in ents_tokens and len(w) >= 3]
            word_doc_count.update(set(filtered_words)) # count number of docs words appear in 

            no_space_bookname = book.replace(' - ', '__').replace(' ', '_') # to avoid spaces in the bookname column of our tsv
            out.append({'doc_id': doc_num, 'book_filename': no_space_bookname, 'text': filtered_words}) 
            doc_num += 1
        
    # remove words used by more than 25% of documents in corpus
    ceiling = math.ceil(0.25*doc_num)
    # remove words occurring in fewer than five books 
    floor = 5

    with open(os.path.join(out_dir, passage_type + '_all_input_docs.tsv'), 'w', encoding='utf8') as outfile: 
        writer = csv.writer(outfile, delimiter='\t')
        for row in out: 
            filtered_words = row['text']
            filtered_words = [w for w in filtered_words if (word_doc_count[w] < ceiling and word_doc_count[w] > floor)]
            writer.writerow([row['doc_id'], row['book_filename'], ' '.join(filtered_words)]) 
            
    with open(os.path.join(out_dir, passage_type + '_doc_num_passage_id.json'), 'w') as outfile: 
        json.dump(doc_num_passage_id, outfile)