# Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes

**Authors**: Li Lucy Camilla Griffiths, Sarah Levine, Jennifer L. Eberhardt, Dorottya Demszky, David Bamman

## Abstract

Conventional bag-of-words approaches for topic modeling, like latent Dirichlet allocation (LDA), struggle with literary text. Literature challenges lexical methods because narrative language focuses on immersive sensory details instead of abstractive description or exposition: writers are advised to *show, don't tell*. We propose Retell, a simple, accessible topic modeling approach for literature. Here, we prompt resource-efficient, generative language models (LMs) to *tell* what passages *show*, thereby translating narratives' surface forms into higher-level concepts and themes. By running LDA on LMs' retellings of passages, we can obtain more precise and informative topics than by running LDA alone or by directly asking LMs to list topics. To investigate the potential of our method for cultural analytics, we compare our method's outputs to expert-guided annotations in a case study on racial/cultural identity in high school English language arts books. 

**Accepted to Findings of ACL 2025**

**Paper link**: [PDF](https://lucy3.github.io/assets/tell_dont_show.pdf)

## Running our approach

Our approach is more conceptual than technical. That is, we argue that classic bag-of-words topic modeling works better once literary text has undergone some abstraction (clinking of silverware and glass at a table → dinner scene).

All you need is to input the following prompt into a language model: 

```
prompt = "In one paragraph, " + abstraction_type + " the following book excerpt for a literary scholar analyzing narrative content. Do not include the book title or author’s name in your response; " + abstraction_type + " only the passage.\n\nPassage:\n" + passage + "\n"
```

Here, `abstraction_type` may be some instructive verb, e.g. "summarize" or "describe". We also use the system prompt `"You are a helpful assistant; follow the instructions in the prompt."`

Then, [download Mallet](https://github.com/mimno/Mallet/releases) and apply topic modeling on models' outputs. 

This repo includes code for: 
- calling different language models (`run_lm.py`)
- preprocessing text, e.g. language models' retellings or the original passages, for LDA (`preprocess_text_for_lda.py`)
- calling Mallet (`topic_modeling.sh` + `topic_modeling_helper.sh`)

Note that in the paper we use a March 27, 2024 version of [TopicGPT](https://github.com/chtmp223/topicGPT) as our "TopicGPT-lite" baseline, with modified/simplified prompts so that "small" LMs can cooperate. 

## Evaluation data

Our evaluation data consists of 11,666 gold-labeled passages and a same-sized sample of random passages, yielding 23,332 passages in total. The addition of random passages ensures that we have enough data to yield good topics with LDA-based approaches. The data we release has 18,041 deduplicated `(passage, theme)` pairs to match our evaluation process. Duplicates arise from multiple scraped quote-theme pairs mapping onto the same passage-theme pair, since a passage may contain multiple or overlapping quotes. So, the number of *unique* passages within each theme are a little lower than the post-scraping counts reported in Table 7 of our paper's Appendix. 

Some of the data we evaluated on are copyrighted, and we purchased and digitized them ourselves. Thus, `book_passages_with_labels.json` only includes minimal information that could support your reconstruction of our data: passage starts, ends, and length. Length counts the number of white-space delimited tokens in a passage, and it is an approximation, as a considerable segment of our books were OCR'd, so digitization error could slightly distort this value. "Tags" are the original labels scraped from online sources, while "themes" derive from our remapping of tags into broader themes. Our paper evaluates topic modeling using the latter labels. 

```
{ 
    book_passage_id: 
    {
        passage_start: "",
        passage_end: "", 
        passage_len: "", 
        tags: {'Sparknotes': [], 'Litcharts': [], 'Goodreads': []}, 
        themes: {'Sparknotes': [], 'Litcharts': [], 'Goodreads': []}, 
    }
}
```
