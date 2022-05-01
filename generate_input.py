from email.policy import default
import pandas as pd
import numpy as np
import os
import re
import copy
import pickle
import itertools
import argparse
# from konlpy.tag import Mecab
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from nltk import sent_tokenize
from tensorflow.keras.preprocessing import sequence 
from sklearn.model_selection import train_test_split

# Replace NAN to ""
def delete(text):
    if type(text) == float : 
        text = ""
    return text

# creating word2index
def create_word2index(data,number,eng,chi):
	token_set = set() 
	word2index = dict()
	for i in range(data.shape[0]): 
		for word in data.loc[i,'body'].split(): 
			token_set.add(word) 
		for word in data.loc[i,'title'].split(): 
			token_set.add(word) 
	counter = 0 
			

	print("**********Creating word2index**********")
	#word2index = {"<OOV>":1,"<숫자>":2,"<영어>":3,'<한자>':4} # 1 for OOV, 2 for number, 3 for english, 4 for chinese
	#index = 5
	for word in token_set: 
		word2index[word] = counter 
		counter = counter+1
	
		
	print("**********DONE!!!**********")

	return word2index

# creating embedding matrix
def create_embedding_matrix(word2index):
	print ("**********Creating word embedding matrix**********") 
	file1 = 'words.word2vec'
	glove2word2vec('glove.6B.100d.txt', file1)
	print("getting keyed vectors")
	model = KeyedVectors.load_word2vec_format(file1, binary=False)
	MAX_VOC = len(word2index)+1 # +1 for padding
	print("MAX_VOC : ",MAX_VOC)
	EMBEDDING_DIM = 100
	embedding_matrix = np.zeros(shape=(MAX_VOC, EMBEDDING_DIM), dtype='float32') # zero vector (index=0)
	for key in word2index:
		try: 
			vec = model.get_vector(key) 
			embedding_matrix[word2index[key]] = np.array(list(vec))
		except KeyError:
			embedding_matrix[word2index[key]] = np.random.uniform(-0.25, 0.25, EMBEDDING_DIM)
	print("**********DONE!!!**********")
	print("**********Embedding Matrix**********")
	print("Embedding matrix shape : {}".format(embedding_matrix.shape))

	return embedding_matrix

# tokenize
def tokenize(data,word2index,tokenizer,number,eng,chi):
	# Sent tokenize body and define body tokenizer
	data["body"] = data["body"].apply(sent_tokenize)
	sent_tokenizer = lambda x : list(map(tokenizer.pos,x))

	# POS tokenize data
	data["title"]=data["title"].apply(tokenizer.pos)
	#data["subtitle"]=data["subtitle"].apply(tokenizer.pos)
	data["body"]=data["body"].apply(sent_tokenizer)
	#data["caption"]=data["caption"].apply(tokenizer.pos)


	for i in range(data.shape[0]):
		for index,(token,pos) in enumerate(data["title"][i]):
			if number.match(token) is not None:
				data["title"][i][index] = ("<숫자>","special_token")
			if eng.match(token) is not None:
				data["title"][i][index] = ("<영어>","special_token")
			if chi.match(token) is not None: 
				if pos == 'SH':
					data["title"][i][index] = ("<한자>","special_token")
				if pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: 
					temp  = f'<{token}_{pos}>'
					data["title"][i][index] = (temp,"special_token")
		# for index,(token,pos) in enumerate(data["subtitle"][i]):
		# 	if number.match(token) is not None:
		# 		data["subtitle"][i][index] = ("<숫자>","special_token")
		# 	if eng.match(token) is not None:
		# 		data["subtitle"][i][index] = ("<영어>","special_token")
		# 	if chi.match(token) is not None: 
		# 		if pos == 'SH':
		# 			data["subtitle"][i][index] = ("<한자>","special_token")
		# 		if pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: 
		# 			temp  = f'<{token}_{pos}>'
		# 			data["subtitle"][i][index] = (temp,"special_token")
		for index,sent in enumerate(data["body"][i]):
			for idx,(token,pos) in enumerate(data["body"][i][index]):
				if number.match(token) is not None:
					data["body"][i][index][idx] = ("<숫자>","special_token")
				if eng.match(token) is not None:
					data["body"][i][index][idx] = ("<영어>","special_token")
				if chi.match(token) is not None: 
					if pos == 'SH':
						data["body"][i][index][idx] = ("<한자>","special_token")
				if pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: 
					temp  = f'<{token}_{pos}>'
					data["body"][i][index][idx] = (temp,"special_token")
		# for index,(token,pos) in enumerate(data["caption"][i]):
		# 	if number.match(token) is not None:
		# 		data["caption"][i][index] = ("<숫자>","special_token")
		# 	if eng.match(token) is not None:
		# 		data["caption"][i][index] = ("<영어>","special_token")
		# 	if chi.match(token) is not None: 
		# 		if pos == 'SH':
		# 			data["caption"][i][index] = ("<한자>","special_token")
		# 	if pos in ['SF', 'SE', 'SSO', 'SSC', 'SC', 'SY']: 
		# 		temp  = f'<{token}_{pos}>'
		# 		data["caption"][i][index] = (temp,"special_token")		
	return data

def find_index(tup):
	global word2index
	if tup[0] in word2index:
		return word2index[tup[0]]
	else :
		return 1


parser = argparse.ArgumentParser()
## Required parameters 
parser.add_argument("--trainingSamples" , type=int ,default=100) 
parser.add_argument("--testingSamples" , type=int,default = 100)
parser.add_argument("--path", type=str)
parser.add_argument("--data_set", type=str)	
parser.add_argument("--w2i", action="store_true")
parser.add_argument("--emb", action="store_true")
parser.add_argument('--max_tit', type=int, default=29)
parser.add_argument('--max_sub', type=int, default=114)
parser.add_argument('--max_body', type=int, default=35)
parser.add_argument('--max_sent', type=int, default=12)
parser.add_argument('--max_cap', type=int, default=24)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--valid_size', type=float, default=0.1)
args = parser.parse_args()
word2index = [] 

def geninps():


	# args.path = "C:\User\msgto\CS529final\FNC_Bin_Train.csv" 
	data = pd.read_csv("FNC_Bin_Train.csv")
	data = data.rename(columns = {'articleBody':'body','Headline':'title','Stance':'label'}) 
	print(data.columns) 
	print(data.shape) 
	print(args.trainingSamples)
	data = data[['title','body','label']][:args.trainingSamples] 
	
	testdata = pd.read_csv("FNC_Bin_Test.csv")
	testdata = testdata.rename(columns = {'articleBody':'body','Headline':'title','Stance':'label'}) 
	print(testdata.columns) 
	print(testdata.shape) 
	print(args.testingSamples)
	testdata = testdata[['title','body','label']][:args.testingSamples]
	# load mecab tokenizer
	#tokenizer = Mecab()
	# declare regular expression variables
	number = re.compile('[0-9]+')
	eng = re.compile('[a-zA-Z]+')
	chi  = re.compile('[一-龥]+')
	# create word2index
	global word2index	
	word2index = create_word2index(data,number,eng,chi)
	word2index_path = "word2ind.pk" 
	with open(word2index_path,'wb') as fw:
		pickle.dump(word2index, fw)

	print("embedding begins")
	embedding_matrix = create_embedding_matrix(word2index) 
	np.save("embedding_mat",embedding_matrix)
	print("embedding completed")
	data['body'] = data['body'].apply(itertools.chain)
	testdata['body'] = testdata['body'].apply(itertools.chain)

	indexing = lambda sent : np.array(list(map(find_index,sent)))
	body_indexing = lambda sent_list : np.array(list(map(indexing,sent_list)))
	data["title"] = data["title"].apply(indexing)
	#data["subtitle"] = data["subtitle"].apply(indexing)
	data["body"] = data["body"].apply(body_indexing)
	#data["caption"] = data["caption"].apply(indexing)

	# pad max_len sentences in body
	data["body"] = data["body"].apply(lambda row : sequence.pad_sequences(row,maxlen=args.max_body,padding='post', truncating='post')) 



	testdata["title"] = testdata["title"].apply(indexing)
	#testdata["subtitle"] = testdata["subtitle"].apply(indexing)
	testdata["body"] = testdata["body"].apply(body_indexing)
	#testdata["caption"] = testdata["caption"].apply(indexing)

	# pad max_len sentences in body
	testdata["body"] = testdata["body"].apply(lambda row : sequence.pad_sequences(row,maxlen=args.max_body,padding='post', truncating='post'))

	# train, dev, test split
	x_train, x_valid, y_train, y_valid = train_test_split(data[["title","body"]], data[['label']], test_size=args.test_size, shuffle=True, stratify=data[["label"]], random_state=486)
	#x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=args.valid_size, stratify=y_train[["label"]], random_state=2018)
	x_test,y_test = testdata[["title","body"]],testdata[["label"]]
	# truncating & padding
	train_title = sequence.pad_sequences(x_train["title"],maxlen=args.max_tit,padding='post', truncating='post')
	#train_subtitle = sequence.pad_sequences(x_train["subtitle"],maxlen=args.max_sub,padding='post', truncating='post')
	train_body= sequence.pad_sequences(x_train["body"],maxlen=args.max_sent,padding='post', truncating='post')
	#train_caption = sequence.pad_sequences(x_train["caption"],maxlen=args.max_cap,padding='post', truncating='post')
	valid_title = sequence.pad_sequences(x_valid["title"],maxlen=args.max_tit,padding='post', truncating='post')
	#valid_subtitle = sequence.pad_sequences(x_valid["subtitle"],maxlen=args.max_sub,padding='post', truncating='post')
	valid_body= sequence.pad_sequences(x_valid["body"],maxlen=args.max_sent,padding='post', truncating='post')
	#valid_caption = sequence.pad_sequences(x_valid["caption"],maxlen=args.max_cap,padding='post', truncating='post')
	test_title = sequence.pad_sequences(x_test["title"],maxlen=args.max_tit,padding='post', truncating='post')
	#test_subtitle = sequence.pad_sequences(x_test["subtitle"],maxlen=args.max_sub,padding='post', truncating='post')
	test_body= sequence.pad_sequences(x_test["body"],maxlen=args.max_sent,padding='post', truncating='post')
	#test_caption = sequence.pad_sequences(x_test["caption"],maxlen=args.max_cap,padding='post', truncating='post')
	y_train = y_train['label'].values
	y_valid = y_valid['label'].values
	y_test = y_test['label'].values


	np.save('train_title', train_title)
	#np.save(index_inputs_path+'/train/train_subtitle', train_subtitle)
	np.save('train_body', train_body)
	#np.save(index_inputs_path+'/train/train_caption', train_caption)
	np.save('valid_title', valid_title)
	#np.save(index_inputs_path+'/train/valid_subtitle', valid_subtitle)
	np.save('valid_body', valid_body)
	#np.save(index_inputs_path+'/train/valid_caption', valid_caption)
	np.save('test_title', test_title)
	#np.save(index_inputs_path+'/test/test_subtitle', test_subtitle)
	np.save('test_body', test_body)
	#np.save(index_inputs_path+'/test/test_caption', test_caption)
	np.save('train_label',y_train)
	np.save('valid_label',y_valid)
	np.save('test_label',y_test)