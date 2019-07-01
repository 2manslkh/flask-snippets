import sys
import json
import operator
from collections import defaultdict
from textblob import TextBlob

import nltk
# nltk.download('vader_lexicon')
# nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiAnalyzer = SentimentIntensityAnalyzer()

import neuralcoref
import spacy
import pandas as pd

nlp = spacy.load('en_core_web_md')
neuralcoref.add_to_pipe(nlp)

'''
To Do:
- define opinion segments
- identifying key list of entities (keyword search?)
- convert text:"i'm" to (author)
- classifying aspects
- manage abbreviations (entity normalisation and consolidation)
- spelling correction(?)
'''

def get_coreferences(doc):
	'''
	Obtains dictionary of coreferences, together with the word token indexes
	'''
	coref_output = defaultdict(list)

	#check that there are coreferences
	if bool(doc._.coref_clusters):
		#print(doc._.coref_clusters)
		for cluster in doc._.coref_clusters:
			#identifying antecedents
			#print("Reference: {}".format(cluster.main))
			#print("Start: {}".format(cluster.main.start))
			#print("End: {}".format(cluster.main.end))
			#coref_output[i].append({'start':cluster.main.start, 'end':cluster.main.end})
			for section in cluster.mentions:
				#identifying anaphora, cataphora, coreferring noun phrases
				#outputs index of start and end token
				coref_output[cluster.main.text].append({'start':section.start, 'end':section.end})
	
	return coref_output

def get_reln(doc, nlp):
    
    sent_text = nltk.sent_tokenize(doc.text)
    
    lookup = ["prep","acl","acomp","amod","attr","aux","nsubj","ccomp","csubj","csubjpass","obj","pobj","dobj","iobj","meta","neg","nmod","nn","nounmod","npmod","nsubjpass","nummod","xcomp"]
    
    order = pd.DataFrame([["nsubj",1],["csubj",2],["nsubjpass",3],["csubjpass",4],["det",5],["attr",6],["nn",7],["acl",8],["acomp",9],["amod",10],["aux",101],["neg",102],["advmod",103],["ROOT",104],["advcl",105],["xcomp",106],["ccomp",107],["meta",108],["prep",109],["nummod",110],["obj",1001],["pobj",1001],["dobj",1001],["iobj",1001],["nounmod",1001],["npmod",1001],["nmod",1001]])
    
    final_mapping = []
    for sentence in sent_text:
#        if coref==1:
#            doc1=nlp(sentence)
#            doc1._.coref_clusters
#            doc1 = doc1._.coref_resolved
#            doc2 = nlp(doc1)
#        else:
#            doc2=nlp(sentence)
        doc2=nlp(sentence)  
        sent_order=pd.DataFrame([[token.text,token.dep_,token.i] for token in doc2])
        sent_order = sent_order.merge(order, left_on=1, right_on=0,how='left')
        sent_order = sent_order.fillna(0)
        a = [[token.i,token.text, token.dep_, token.tag_, token.head.text,[child for child in token.children],
              [child.dep_ for child in token.children],[child.i for child in token.children]] for token in doc2]
        df=pd.DataFrame(a) 

        # remove all other tags and verbs in the child list except for the one in the allowed list - lookup   
        for index, row in df.iterrows():
            if (len(row[7])!=0):
                del_list=[]
                for num,item in enumerate(row[6]):
                    if item not in lookup:
                        del_list.append(num)
                for index in sorted(del_list, reverse=True):
                    del row[5][index]
                    del row[6][index]
                    del row[7][index]
        
        # bring together children that are required to form a sentence            
        for index, row in df.iterrows():
            if (len(row[7])!=0):
                for num,item in enumerate(row[7]):
                    if (len(df.iloc[item,7])!=0):
                        row[5].extend(df.iloc[item,5])
                        row[6].extend(df.iloc[item,6])
                        row[7].extend(df.iloc[item,7])
    #    for index, row in df.iterrows():
    #        row[7].sort()
        # manage conjunctions
         # get root relation for building dependent trees
    #    y = df.index[df[2] == "ROOT"].tolist()[0]
    #    dep1 = []
    #    dep2 = [y]
    #    for ind, item in enumerate(df.iloc[y,6]):
    #        if item[1:4] == "obj" or item == "xcomp":
    #            dep1.append(item)
    #            dep2.append(df.iloc[y,7][ind])
    #    dep2.sort()
    #    
    #    for index, row in df.iterrows():
    #        if row[2]=='conj':
    #            row[7].extend(dep2)
        
        # now bring the parent also in
        for index, row in df.iterrows():
            if (len(row[7])!=0):
                row[7].append(index)
                row[5].append(df.iloc[index,1])
                row[6].append(df.iloc[2,3][:2])
        
        # keep only items with subject and verb        
        for index, row in df.iterrows():
            if (len(row[7])!=0):
                if any("subj" in s for s in row[6]) or any("conj" in s for s in row[6]):
                    if any("VB" or "ROOT" or "xcomp" in s for s in row[6]):
                        continue
                    else:
                        row[7].clear()
                else:
                    row[7].clear()
        
        item_in_maps=[]
        for index, row in df.iterrows():
            if (len(row[7])!=0):
                subj=""
                obj=""
                for num,item in enumerate(row[7]):
                    p = df.iloc[item,1]
                    q = df.iloc[item,2]
                    if q[1:5]=="subj" and subj=="":
                         subj = p
    #                elif q[-4:]=="conj" and df.iloc[item,2][:2]=="NN":
    #                    subj = p
                    if q[-3:]=="obj":
                         obj = p
                    elif q[-5:]=="acomp" and obj == "":
                        obj = p
                if obj != "":
                    obj_pos = df.iloc[df.index[df[1] == obj].tolist()[0],0]
                if obj=="":
                    obj="movie"
                    obj_pos = max(df[0])+1
                item_in_maps.append([obj,obj_pos,subj,df.iloc[df.index[df[1] == subj].tolist()[0],0]])

        sent = []            
        for index, row in df.iterrows():
            if (len(row[7])!=0):
                temp={}
                for num,item in enumerate(row[7]):   
                    temp[df.iloc[item,1]]=sent_order.iloc[(sent_order.index[sent_order['0_x'] == df.iloc[item,1]].tolist()[0]),5]
                temp = sorted(temp.items(), key=operator.itemgetter(1))
                temp = [item[0] for item in temp]
                temp.append(".#")
                temp = " ".join(temp)
                temp = temp.split("#")
                temp = [each.replace(" .",".") for each in temp]
                temp.remove('')
                temp = [each.strip() for each in temp]
                sent.append(temp)
                
        
        senti_score1 = [sentiAnalyzer.polarity_scores(each[0])['compound'] for each in sent]
        senti_score2 = [TextBlob(each[0]).sentiment.polarity for each in sent]
        senti_score = []
        for i in range(len(senti_score1)):
            senti_score.append((senti_score1[i]+senti_score2[i])/2)
        #subjectivity_score = [TextBlob(each[0]).sentiment.subjectivity for each in sent]
#        for each in sent:
#            print(each[0])
#            print(TextBlob(each[0]).sentiment.subjectivity)
        each_mapping = []
        for index, item in enumerate(item_in_maps):
            each_mapping.append([item[0],item[1],item[2],item[3],senti_score[index]])
        final_mapping.extend(each_mapping)    
    
    final_mapping = pd.DataFrame(final_mapping)
    for index, row in final_mapping.iterrows():
        if row[0]==row[2]:
            final_mapping = final_mapping.drop([index])
    final_mapping = final_mapping.groupby([0,1,2,3])[4].mean().reset_index()
    return (final_mapping)

def get_sentiments_entity_relationships(doc,json_output,nlp):
    reln_out = get_reln(doc, nlp)

	#analysis_output_all = defaultdict(list)
    for sent in doc.sents:
		#identify opinion segments (if any) -> keyword driven currently
		#for word in sent...
		#get sentiment score for each sentence
        analysis_output = defaultdict(list)
        analysis_output['sentiment'] = sentiAnalyzer.polarity_scores(sent.text)
		#print(json.dumps(analysis_output))
		
		#implement grammar rules to tease out entity-sentiment relationships
        for token in sent:
			#print(token.text, token.pos_, token.dep_)
            analysis_output['dependency'].append({'text':token.text, 'pos':token.pos_, 'dep':token.dep_})
		
		#apply grammar rules here

		#initialize
        subject = ""
        object = ""
        subject_idx = {"start":-1, "end":-1}
        object_idx = {"start":-1, "end":-1}
        temp_object = ""
        temp_object_idx = {"start":0, "end":0}
        appos_mod = ""

        for count, token in enumerate(analysis_output['dependency']):
			#print("token:{}, dep:{}".format(token['text'], token['dep']))
			
			### Identifying and extracting subjects ###
			#Base case: plain old nsubj, check that part-of-speech is not adjective
            if (token['dep'] == 'nsubj') and (token['pos'] != "ADJ"):
                subject = token['text']
                subject_idx["start"] = count
                subject_idx["end"] = count + 1
				
				#Rule 1: Check if it is a compound entity
                if count-1 >= 0:
					#check not the first token
					#Rule 1c: Check for single compounded nsubj
                    token_minus_1 = analysis_output['dependency'][count-1]
                    if token_minus_1['dep']=='compound' and token['dep']=='nsubj':
                        subject = " ".join([token_minus_1['text'], token['text']])
                        subject_idx["start"] = count - 1

                if count-2 >= 0:
                    token_minus_1 = analysis_output['dependency'][count-1]
                    token_minus_2 = analysis_output['dependency'][count-2]
					# token_minus_3 = analysis_output['dependency'][count-3]
					#Rule 1a: If nsubj is preceded by possessive and case dependencies, extract compound, poss, case and nsubj
					#check that there are at least 3 more token behind
                    if token_minus_2['dep']=='compound' and token_minus_1['dep']=='poss' and token['dep']=='nsubj':
                        subject = " ".join([token_minus_2['text'], token_minus_1['text'], token['text']])
                        subject_idx["start"] = count - 2
						
					#Rule 1b: Check for multiple compounded nsubj
                    elif token_minus_2['dep']=='compound' and token_minus_1['dep']=='compound' and token['dep']=='nsubj':
                        subject = " ".join([token_minus_2['text'], token_minus_1['text'], token['text']])
                        subject_idx["start"] = count - 2
				#print("Subject: {}".format(subject))

			### Identifying and extracting objects ###
			#Base case: plain old dobj/ pobj (extracting direct or passive objects)
            if ((token['dep'] == 'dobj') or (token['dep'] == 'pobj')) and (token['pos'] != "ADJ"):
                object = token['text']
                object_idx["start"] = count
                object_idx["end"] = count + 1

				#Rule 2: Check if it is a compound entity
                if count-1 >= 0:
					#check not the first token
                    token_minus_1 = analysis_output['dependency'][count-1]
                    if token_minus_1['dep']=='compound' and token['dep']=='dobj':
                        object = " ".join([token_minus_1['text'], token['text']])
                        object_idx["start"] = count - 1
                    elif token_minus_1['dep']=='amod' and token['dep']=='dobj':
                        object = " ".join([token_minus_1['text'], token['text']])
                        object_idx["start"] = count - 1

                if count-2 >= 0:
                    token_minus_1 = analysis_output['dependency'][count-1]
                    token_minus_2 = analysis_output['dependency'][count-2]
					# token_minus_3 = analysis_output['dependency'][count-3]

					#Rule 1a: If dobj or pobj is preceded by possessive and case dependencies, extract compound, poss, case and nobj
					#check that there are at least 3 more token behind
                    if token_minus_2['dep']=='compound' and token_minus_1['dep']=='poss' and ((token['dep']=='dobj') or (token['dep']=='pobj')):
                        object = " ".join([token_minus_2['text'], token_minus_1['text'], token['text']])
                        object_idx["start"] = count - 2
						
					#Rule 1b: Check for multiple compounded nsubj
                    elif token_minus_2['dep']=='compound' and token_minus_1['dep']=='compound' and ((token['dep']=='dobj') or (token['dep']=='pobj')):
                        object = " ".join([token_minus_2['text'], token_minus_1['text'], token['text']])
                        object_idx["start"] = count - 2

					#Rule 1c: Check for pobj entities separated by preposition
                    elif ((token_minus_2['dep']=='dobj') or (token_minus_2['dep']=='pobj')) and token_minus_1['dep']=='prep' and ((token['dep']=='dobj') or (token['dep']=='pobj')):
                        object = " ".join([token_minus_2['text'], token_minus_1['text'], token['text']])
                        object_idx["start"] = count - 2

			#Rule 2: temporarily store prepositional objects. Direct objects
			# This only stores the latest pobj 
			# if token['dep']=='pobj':
			# 	temp_object = token['text']
			# 	temp_object_idx["start"] = count
			# 	temp_object_idx["end"] = count + 1

			#Rule 3: Identifying appositional modifiers (assume to modify noun subjects only)
            if token['dep']=='appos':
                appos_mod = token['text']
                object_idx["start"] = count
                object_idx["end"] = count + 1
                if count-1 >= 0:
					#check not the first token
                    token_minus_1 = analysis_output['dependency'][count-1]
                    if token_minus_1['dep']=='compound' and token['dep']=='appos':
                        appos_mod = " ".join([token_minus_1['text'], token['text']])
                        object_idx["start"] = count - 1
				# print(appos_mod)
			
		# Additional processing after iterating through the whole sentence
		# Following from rule #2, if no direct object present, use pobj
        if object == "" and temp_object != "":
            object = temp_object
            object_idx = temp_object_idx
		
		#Rule 3: Sentences without subjects (Imperative/Axiomatic expression)
        if object != "" and subject == "":
            subject = "movie"
		
		#Rule 4: Sentences without objects ()
        if object == "" and subject != "":
            object = subject
            subject = "movie"
            object_idx = subject_idx
            subject_idx = {"start":-1, "end":-1}
		
		#Rule 5: Sentences without objects and subjects
        if object == "" and subject == "":
            continue
		#Rule 6: Sentences without objects or subjects
        elif object == "" or subject == "":
            continue
        else:
            json_output['parsed'].append(analysis_output)
            if appos_mod != "":
                subject = "-".join([subject, appos_mod])
            
    
    for _, row in reln_out.iterrows():
        subject = row[2]
        object = row[0]
        subject_idx = {"start":row[3], "end":row[3]+1}
        object_idx = {"start":row[1], "end":row[1]+1}
        senti_score = row[4]
        json_output['relationship'].append({'subject':subject, 'subject_idx':subject_idx,'object':object, 'object_idx':object_idx,'sentiment':senti_score})

	#tokenize and number each token
	#index starts with 0
	#tokenizer = Tokenizer(nlp.vocab) #does not tokenize punctuation, only vocab
	#print([token.text for token in doc])
    for idx, token in enumerate(doc):
        json_output['raw'].append({'idx':idx, 'token':token.text})
	
    return(json_output)

def create_json_output_single_file(inp):
    '''
    This function parses a single document at a time.
    Note: assumes data is clean already
    Input: takes in a single string to be parsed
    Output: generates a JSON file that contains the raw, dependency parsing results and co-references for visualisation in javascript
    '''
    # Initialises the dictionary with an empty list as value

    json_output_original = defaultdict(list)
    json_output_resolved = defaultdict(list)

    doc = nlp(inp)
    
    json_output_original['coreference'] = get_coreferences(doc)
    json_output_original = get_sentiments_entity_relationships(doc, json_output_original, nlp)

    input_resolved = doc._.coref_resolved
    doc_resolved = nlp(input_resolved)
    json_output_resolved = get_sentiments_entity_relationships(doc_resolved, json_output_resolved, nlp)
    json_output = {"original":json_output_original, "resolved":json_output_resolved, "original_content": inp, "resolved_content":input_resolved}
    return json_output

if __name__ == "__main__":
	print(create_json_output_single_file("hello world, it is a good day today"))

	# print(create_json_output_single_file(sys.argv[2]))

