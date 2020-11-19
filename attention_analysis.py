# -*-coding:utf-8-*-
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=FutureWarning)
import copy
import os
import pickle
import time

import numpy
import pyconll
import torch
import transformers
from transformers import BertModel, BertTokenizer

def timer(func):
	def wrapper(*args,**kw):
		start = time.time()
		func(*args,**kw)
		end = time.time()
		duration = end - start
		print("timer>>>> "+str(duration))
	return wrapper 

def main():
	load_config()
	load_model()
	load_treebank(corpus_filename,treebank_wd)
	print("main:>>>> BERT version"+model_version)
	print("main:>>>> Treebank path:"+treebank_wd)
	print("main:>>>> Treebank filename:"+corpus_filename)
	print("main:>>>> Main function: precession_all()")
	print("main:>>>> Typed dependenies to be processed: "+str(relist))
	print("main:>>>> For sentence-wise analysis，run debugger()")
	return


def load_config():
	global model_version,treebank_wd,corpus_filename,relist,attention_matrix_wd,head_prec_wd, wordified_matrix_wd,precession_filename,pickle_matrix_wd, pickle_filename
	model_version = 'bert-base-uncased'
	
	default_treebank_wd = "D:\\thehow\\tasks\\attention_analysis\\treebank\\64\\sd_conll\\"
	print("load_config:>>>> ",end="")
	print(" current treebank path： ",end="")
	print(default_treebank_wd)	
	treebank_wd = input("load_config:>>>>  treebank path: ")
	if treebank_wd =="":
		treebank_wd = default_treebank_wd
	else:
		print("load_config:>>>> current: ",end="")
		print(treebank_wd)

	default_corpus_filename = "64_sd.conll"
	print("load_config:>>>> ",end="")
	print("treebank filename： ",end="")
	print(default_corpus_filename)	
	corpus_filename = input("load_config:>>>> treebank filename: ")
	if corpus_filename == "":
		corpus_filename = default_corpus_filename
	else:
		print("load_config:>>>> current: ",end="")
		print(corpus_filename)
			
	default_attention_matrix_wd ="D:\\thehow\\tasks\\attention_analysis\\output\\attention_matrix\\"
	print("load_config:>>>> ",end="")
	print("attention_matrix path： ",end="")
	print(default_attention_matrix_wd)	
	attention_matrix_wd = input("load_config:>>>> Attention-head Matrix path： ")
	if attention_matrix_wd =="":
		attention_matrix_wd =default_attention_matrix_wd
	else:
		print("load_config:>>>> current: ",end="")
		print(attention_matrix_wd)
			
	default_head_prec_wd = "D:\\thehow\\tasks\\attention_analysis\\output\\head_rel_prec\\"
	print("load_config:>>>> ",end="")
	print("Head Precession path： ",end="")
	print(default_head_prec_wd)	
	head_prec_wd = input("load_config:>>>> Head Precession path： ")
	if head_prec_wd == "":
		head_prec_wd = default_head_prec_wd
	else:
		print("load_config:>>>> current: ",end="")
		print(head_prec_wd)

	default_wordified_matrix_wd = "D:\\thehow\\tasks\\attention_analysis\\output\\WORDIFIED_MATRIX\\"
	print("load_config:>>>> ",end="")
	print("wordified Matrix path： ",end="")
	print(default_wordified_matrix_wd)	
	wordified_matrix_wd = input("load_config:>>>> wordified Matrix path path： ")
	if wordified_matrix_wd == "":
		wordified_matrix_wd = default_wordified_matrix_wd
	else:
		print("load_config:>>>> current: ",end="")
		print(wordified_matrix_wd)

	default_precession_filename = str(int(time.time()))+"PREC_SD_64_WORD_BI.txt"
	print("load_config:>>>> ",end="")
	print("Precession Filename： ",end="")
	print(default_precession_filename)	
	precession_filename = input("load_config:>>>> Precession Filename： ")
	if precession_filename == "":
		precession_filename = default_precession_filename
	else:
		print("load_config:>>>> current: ",end="")
		print(precession_filename)

	default_pickle_matrix_wd = "D:\\thehow\\tasks\\attention_analysis\\treebank\\64\\paper_pkl\\"
	print("load_config:>>>> ",end="")
	print("Pickled Matrix path： ",end="")
	print(default_pickle_matrix_wd)	
	pickle_matrix_wd = input("load_config:>>>> Pickled Matrix path： ")
	if pickle_matrix_wd == "":
		pickle_matrix_wd = default_pickle_matrix_wd
	else:
		print("load_config:>>>> current: ",end="")
		print(pickle_matrix_wd)

	default_pickle_filename = "dev_attn.pkl"
	print("load_config:>>>> ",end="")
	print("pickle filename： ",end="")
	print(default_pickle_filename)	
	pickle_filename = input("load_config:>>>> pickle filename： ")
	if pickle_filename == "":
		pickle_filename = default_pickle_filename
	else:
		print("load_config:>>>> current: ",end="")
		print(pickle_filename)

	relist = ['pobj','poss','det','dobj','auxpass']
	return

def load_model():
	global MODEL,TOKENIZER
	do_lower_case = True
	MODEL = BertModel.from_pretrained(model_version, output_attentions=True)
	TOKENIZER = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
	return

def load_treebank(corpus_filename, treebank_wd):
	global CORPUS
	filepath = treebank_wd+corpus_filename
	CORPUS = pyconll.load_from_file(filepath)
	return CORPUS

def preproc_one_sent(sent_id):
	sent_ud = get_sent_ud(sent_id)
	sent_raw = get_sent_raw(sent_ud)
	sent_bert = get_sent_bert(sent_raw)
	id_conv_table = get_id_conv_table(sent_ud,sent_bert)
	dep_head_table = [(int(token.id),int(token.head)) for token in sent_ud]
	return {'sent_id':sent_id,'sent_raw':sent_raw,'sent_bert':sent_bert,'sent_ud':sent_ud,'id_conv_table':id_conv_table,'dep_head_table':dep_head_table}

def one_head_one_sent_one_rel_precession(layer,head,sent_id,rel):
	BERT_ATTENTION_MATRIX = save_one_head_one_sent_bert_attention_matrix(layer,head,sent_id,0)
	WORDIFIED_MATRIX = save_one_head_one_sent_wordified_matrix(layer,head,sent_id,0)
	rel_info = one_sent_rel_info(layer,head,sent_id,rel)
	one_sent_rel_value_sum = rel_info[0]
	one_sent_rel_count_sum = rel_info[1]
	if one_sent_rel_count_sum == 0:
		precession = 0
	else:
		precession = one_sent_rel_value_sum/one_sent_rel_count_sum
	return {'precession':precession,'value_sum':one_sent_rel_value_sum,'count_sum':one_sent_rel_count_sum}

def one_head_all_sent_one_rel_precession(CORPUS,layer,head,rel):
	start = time.clock()
	sent_rel_value_list = []
	sent_rel_count_list = []
	
	for sent_id in range(len(CORPUS)):
		res_container = one_head_one_sent_one_rel_precession(layer,head,sent_id,rel)
		sent_rel_value_list.append(res_container['value_sum'])
		sent_rel_count_list.append(res_container['count_sum'])
	
	corpus_sum = sum(sent_rel_value_list)
	corpus_len = sum(sent_rel_count_list)
	
	if corpus_len == 0:
		corpus_prec = 0
	else:
		corpus_prec = corpus_sum/corpus_len
	end = time.clock()
	duration = end-start
	print("one_head_all_sent_one_rel_precession>>>> Duration: "+str(duration))
	print("one_head_all_sent_one_rel_precession>>>> Layer: "+str(layer)+" "+"Head: "+str(head)+" "+"Relation: "+rel)
	print("one_head_all_sent_one_rel_precession>>>> Precession: "+ str(corpus_prec))
	return (layer,head,rel,corpus_prec)


def precession_all():
	filename = head_prec_wd + precession_filename
	res_prec_file = open(filename, mode='a+',encoding = 'utf-8')
	for rel in relist:
		for layer in range(12):
			for head in range(12):
				res = one_head_all_sent_one_rel_precession(CORPUS,layer,head,rel)
				res_prec_file.write(str(res))
				res_prec_file.write('\n')
				res_prec_file.flush()
	res_prec_file.close()
	return

def get_sent_ud(sent_id):
	sent_ud = CORPUS[sent_id]
	return sent_ud

def get_sent_raw(sent_ud):
	sent_raw_lst = []
	punct = [',','.','?','!']
	sent_raw = ""
	for token in sent_ud:
		sent_raw_lst.append(token.form)
	for i in sent_raw_lst:
		if i in punct:
			sent_raw = sent_raw + i
			continue
		else:
			sent_raw = sent_raw + " "+i
	if sent_raw[0] == ' ':
		sent_raw = sent_raw[1:]
	return sent_raw

def get_sent_bert(sent_raw):
	inputs = TOKENIZER.encode_plus(sent_raw, return_tensors='pt', add_special_tokens=True)
	token_type_ids = inputs['token_type_ids']
	input_ids = inputs['input_ids']
	input_id_list = input_ids[0].tolist()
	tokens = TOKENIZER.convert_ids_to_tokens(input_id_list)
	sent_bert = tokens
	return sent_bert

def rel_exist(sent_ud,dep_ordid,head_ordid,rel):
	ref_pair_list = []
	l = 0
	for token in sent_ud:
		ref = (int(token.id),int(token.head))
		ref_pair_list.append(ref)
	if (dep_ordid,head_ordid) in ref_pair_list:
		if sent_ud[dep_ordid-1].deprel == rel:
			l = 1
		else:
			l = 0
	return l

def get_candidate_dep(sent_ud,rel):
	dep_list = []
	for token in sent_ud:
		if token.deprel == rel:
			dep_list.append(int(token.id))
	return dep_list

def get_candidate_head(sent_ud,rel):
	head_list = []
	for token in sent_ud:
		if token.deprel == rel:
			head_list.append(int(token.head))
	head_list = list(set(head_list))
	return head_list

def save_one_head_one_sent_bert_attention_matrix(layer,head,sent_id,write_to_local):
	global BERT_ATTENTION_MATRIX
	sent_raw = preproc_one_sent(sent_id)['sent_raw']
	inputs = TOKENIZER.encode_plus(sent_raw, return_tensors='pt', add_special_tokens=True)
	token_type_ids = inputs['token_type_ids']
	input_ids = inputs['input_ids']
	sent_attention = MODEL(input_ids, token_type_ids=token_type_ids)[-1]
	BERT_ATTENTION_MATRIX = sent_attention[layer][0][head].detach().numpy()
	if write_to_local ==1:
		numpy.set_printoptions(suppress=True,precision=8)
		filename = attention_matrix_wd + "attention_"+"layer-"+str(layer)+"_"+"head-"+str(head)+".csv"
		numpy.savetxt(filename,BERT_ATTENTION_MATRIX,fmt="%10.8f",delimiter=",")
	return BERT_ATTENTION_MATRIX

def most_attend_head(layer,head,sent_id,dep_ordid):
	sent_attention_wordified_matrix = WORDIFIED_MATRIX
	dep_attention_row = sent_attention_wordified_matrix[dep_ordid,:]
	most_attend_head_ordid = numpy.argmax(dep_attention_row).item()
	return most_attend_head_ordid

def most_attend_dep(layer,head,sent_id,head_ordid):
	sent_attention_wordified_matrix = WORDIFIED_MATRIX
	head_attention_col = sent_attention_wordified_matrix[:,head_ordid]
	most_attend_dep_ordid = numpy.argmax(head_attention_col).item()
	return most_attend_dep_ordid

def get_id_conv_table(sent_ud,sent_bert):
	id_conv_table = []
	sent_ud_list = [s.form for s in sent_ud]
	sent_bert_trim = sent_bert[1:-1]
	for i in range(len(sent_ud_list)):
		if sent_bert_trim[i] == sent_ud_list[i].lower():
			id_conv_table.append(i+1)
		elif sent_bert_trim[i] != sent_ud_list[i].lower():
			id_conv_table.append(i+1)
			combine = sent_bert_trim[i]
			while sent_bert_trim[i]!=sent_ud_list[i].lower():
				if sent_bert_trim[i+1].startswith('##'):
					delta = sent_bert_trim[i+1][2:]
				else:
					delta = sent_bert_trim[i+1]
				combine += delta
				sent_bert_trim[i] = combine
				id_conv_table.append(i+1)
				del sent_bert_trim[i+1]
	id_conv_table.insert(0,'[CLS]')
	id_conv_table.append('[SEP]')
	return id_conv_table

def one_sent_rel_info(layer,head,sent_id,rel):
	preproc_res_container = preproc_one_sent(sent_id)
	sent_ud = preproc_res_container['sent_ud']
	sent_bert = preproc_res_container['sent_bert']
	candidate_dep_ordid = get_candidate_dep(sent_ud,rel)
	candidate_head_ordid = get_candidate_head(sent_ud,rel)
	l_list = []

	for dep_ordid in candidate_dep_ordid:
		most_attend_head_ordid = most_attend_head(layer,head,sent_id,dep_ordid)
		l_dep = rel_exist(sent_ud,dep_ordid,most_attend_head_ordid,rel)
		l_list.append(l_dep)
	dep_head_table = preproc_res_container['dep_head_table']
	for head_ordid in candidate_head_ordid:
		most_attend_dep_ordid = most_attend_dep(layer,head,sent_id,head_ordid)
		l_head = rel_exist(sent_ud,most_attend_dep_ordid,head_ordid,rel)
		l_list.append(l_head)
	rel_sum_value = sum(l_list)
	rel_len_value = len(l_list)
	return rel_sum_value,rel_len_value

# def save_one_head_one_sent_wordified_matrix(layer,head,sent_id,write_to_local):
# 	global WORDIFIED_MATRIX
# 	filename = wordified_matrix_wd + "wordified_bert_matrix"+"layer-"+str(layer)+"_"+"head-"+str(head)+".csv"
# 	bert_matrix = BERT_ATTENTION_MATRIX
# 	id_conv_table = preproc_one_sent(sent_id)['id_conv_table']
# 	id_conv_table_1 = copy.deepcopy(id_conv_table)
# 	id_conv_table_2 = copy.deepcopy(id_conv_table)
# 	for i in range(len(id_conv_table_1)-1):
# 			while id_conv_table_1[i+1] == id_conv_table_1[i]:
# 				bert_matrix[:,i] = bert_matrix[:,i]+bert_matrix[:,i+1]
# 				bert_matrix = numpy.delete(bert_matrix,i+1,axis=1)
# 				del id_conv_table_1[i+1]
# 			if i == len(id_conv_table_1)-3:
# 				break
# 	for j in range(len(id_conv_table_2)-1):
# 		cnt = 1
# 		while id_conv_table_2[j+1] == id_conv_table_2[j]:
# 			cnt += 1
# 			bert_matrix[j,:] = bert_matrix[j,:]+bert_matrix[j+1,:]
# 			bert_matrix = numpy.delete(bert_matrix,j+1,axis=0)
# 			del id_conv_table_2[j+1]
# 		bert_matrix[j,:] = bert_matrix[j,:]/cnt
# 		if j == len(id_conv_table_2)-3:
# 			break
# 	WORDIFIED_MATRIX = bert_matrix
# 	if write_to_local == 1:
# 		numpy.savetxt(filename,bert_matrix,fmt="%10.8f",delimiter=",")
# 	return WORDIFIED_MATRIX

def save_one_head_one_sent_wordified_matrix(layer,head,sent_id,write_to_local):
	global WORDIFIED_MATRIX
	save_filename = wordified_matrix_wd + "wordified_bert_matrix"+"layer-"+str(layer)+"_"+"head-"+str(head)+".csv"
	pickle_full_filename = pickle_matrix_wd+pickle_filename
	pkl = open(pickle_full_filename,mode='rb')
	pkl = pickle.load(pkl,encoding='bytes')
	WORDIFIED_MATRIX = pkl[sent_id][b'attns'][layer][head]
	if write_to_local == 1:
		numpy.savetxt(save_filename,WORDIFIED_MATRIX,fmt="%10.8f",delimiter=",")
	return

def debugger(layer,head,sent_id):
	global deb_sent_ud,deb_sent_raw,deb_sent_bert,deb_bert_matrix,deb_wordified_matrix
	deb_sent_ud = get_sent_ud(sent_id)
	deb_sent_raw = get_sent_raw(deb_sent_ud)
	deb_sent_bert = get_sent_bert(deb_sent_raw)
	deb_bert_matrix = save_one_head_one_sent_bert_attention_matrix(layer,head,sent_id,1)
	deb_wordified_matrix = save_one_head_one_sent_wordified_matrix(layer,head,sent_id,1)
	print("bert_matrix, wordified_matrix written to local")
	print(attention_matrix_wd)
	print(wordified_matrix_wd)
	print("-------------------------")
	print(deb_sent_raw)
	print("-------------------------")
	print("bert_matrix.shape = "+ str(deb_bert_matrix.shape))
	print("-------------------------")
	print(str(deb_bert_matrix))
	print("-------------------------")
	print("wordified_matrix.shape = "+ str(deb_wordified_matrix.shape))
	print("-------------------------")
	print(str(deb_wordified_matrix))
	print("-------------------------")

	return

main()
precession_all()