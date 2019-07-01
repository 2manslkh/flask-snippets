import neuralcoref

import spacy
nlp = spacy.load('en_core_web_md')
doc = nlp("hello world, it is a good day today")
neuralcoref.add_to_pipe(nlp)
print("outputsz")   