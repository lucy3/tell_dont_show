# Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes

## Abstract

Conventional bag-of-words approaches for topic modeling, like latent Dirichlet allocation (LDA), struggle with literary text. Literature challenges lexical methods because narrative language focuses on immersive sensory details instead of abstractive description or exposition: writers are advised to *show, don't tell*. We propose Retell, a simple, accessible topic modeling approach for literature. Here, we prompt resource-efficient, generative language models (LMs) to *tell* what passages *show*, thereby translating narratives' surface forms into higher-level concepts and themes. By running LDA on LMs' retellings of passages, we can obtain more precise and informative topics than by running LDA alone or by directly asking LMs to list topics. To investigate the potential of our method for cultural analytics, we compare our method's outputs to expert-guided annotations in a case study on racial/cultural identity in high school English language arts books. 

**Accepted to Findings of ACL 2025**

**Paper link**: TBD

## Running our approach

Since our approach is very simple, all you need is the following instruction prompt: 

```
prompt="In one paragraph, " + abstraction_type + " the following book excerpt for a literary scholar analyzing narrative content. Do not include the book title or author’s name in your response; " + abstraction_type + " only the passage.\n\nPassage:\n" + passage + "\n"
messages = [
  {"role": "system", "content": "You are a helpful assistant; follow the instructions in the prompt."},
  {"role": "user", "content": prompt},
  ]   
```

Here, `abstraction_type` may be some instructive verb, e.g. "summarize", "describe", or "paraphrase". 

This repo also includes code for calling different language models, preprocessing text for LDA, and running Mallet. 
