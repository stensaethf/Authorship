'''
classifier.py
Frederik Roenn Stensaeth
10.09.15

A Python program that, when given a snippet of text, will determine the most
likely correct author (of the given authors) of each sentence in the text.

Tokenization:
- Tokenization has been done as on page 71, with a couple of exceptions. 
  1. The separation of clitics into their full versions has been omitted.
  2. No attempt is made to identify abbreviations other the ones that are
     hardcoded into the abbr dictionary.
'''

import sys
import os.path
import urllib.request
import re
import pickle
import math
import random

def tokenizer(sentence):
	"""
	tokenizer() takes a sentence (string) and tokenizes it according to the
	algorithm for tokenization provided by Jurafsky and Martin in
	'Speech and Language Processing' (2nd) on page 71.

	@params: sentence to be tokenized.
	@return: tokenized sentence.
	"""
	clitic = "'|:|-|'S|'D|'M|'LL|'RE|'VE|N'T|'s|'d|'m|'ll|'re|'ve|n't"
	abbr = {'Co.' : 1, 'Dr.' : 1, 'Jan.' : 1, 'Feb.' : 1, 'Mr.' : 1,
	        'Ms.' : 1, 'Mrs.' : 1, 'Inc.' : 1, 'Mar.' : 1, 'Apr.' : 1,
	        'Aug.' : 1, 'Sept.' : 1, 'Oct.' : 1, 'Nov.' : 1, 'Dec.' : 1}

	tokenized_sent = sentence

	# Put whitespace around separators.
	tokenized_sent = re.sub('([\\?!()\";/|`:])', r' \1 ', tokenized_sent)

	# Put whitespace around commas that are not inside numbers.
	tokenized_sent = re.sub('([^0-9]),', r'\1 , ', tokenized_sent)
	tokenized_sent = re.sub(',([^0-9])', r' , \1', tokenized_sent)

	# Distinguish singlequotes from apostrophes by segmenting off single
	# quotes not preceded by a letter.
	tokenized_sent = re.sub("^\'", r"' ", tokenized_sent)
	tokenized_sent = re.sub("([^A-Za-z0-9])\'", r"\1 '", tokenized_sent)

	# Segment off punctuation from clitics.
	reg = '(' + clitic + ')([^A-Za-z0-9])'
	tokenized_sent = re.sub(reg, r'\1 \2', tokenized_sent)

	# Now periods.
	words = tokenized_sent.split()
	count = -1
	words_new = []
	# Loops over each word and checks if it ends in a period. If it does end
	# with a period we check if it is an abbreviation or a sequence of letters
	# and periods (U.S.)
	for word in words:
		count += 1
		if word[-1] == '.':
			if word in abbr:
				# it is an abbreviation
				words_new.append(word)
			else:
				# not an abbreviation
				if '.' in word[:-1]:
					words_new.append(word)
				else:
					words_new.append(word[:-1])
					words_new.append('.')
		else:
			words_new.append(word)

	tokenized_sent = ' '.join(words_new)

	return tokenized_sent

def openAuthorlist(f):
	"""
	openAuthors() takes a filename and reads the content of the file. The file
	needs to be formated <Name of author> <Url> \n <Name of author> <Url>.
	Returns a list of tuples: (<Name of author> <Url>).

	@params: filename (string).
	@return: List of tuples: (<Name of author> <Url>).
	"""
	assert(os.path.isfile(f))

	# authors = [[name, url], [name, url], [name, url], etc.]
	authors = []
	authors_info = open(f, 'r').read().split('\n')
	for line in authors_info:
		information = line.split(',')
		if len(information) != 2:
			errorMessage()
		authors.append(information)

	return authors

def errorMessage():
	"""
	errorMessage() tells the user that an error has occured. Prints out a
	usage message to the user before exiting the system.

	@params: n/a.
	@return: n/a.
	"""

	print('Error. Invalid command line prompt.')
	print('Usage:')
	print('$ classifier.py -dev <authorlist>')
	print('$ classifier.py -test <authorlist> <testset.txt>')
	print('Authorlist should be of the format:')
	print('authorname1,url1')
	print('authorname2,url2')
	print('etc.')
	sys.exit()

def cleanExtra(content):
	"""
	cleanExtra() takes a string and removes certain characters and features
	from the string. The 'cleaned' string is returned.

	@params: string to be cleaned.
	@return: cleaned string.
	"""
	# Cleans the content, meaning: remove non-letter, number and period
	# characters. Removes the period at the end of the file, as we later
	# want to split on periods to get individual sentences.
	content = re.sub(' [^A-Za-z0-9.]* ', ' ', content)
	content = re.sub('(^ )|( $)', '', content)
	content = re.sub(' .$', '', content)

	return content

def getData(url):
	"""
	getData() takes a url and opens/ reads it. The content of the url
	is cleaned before indivudal sentences are made into a list. This list is
	returned.

	@params: Url (string).
	@return: List of indiviudal sentences (list of strings).
	"""
	# Opens the url and get the content.
	content = urllib.request.urlopen(url).read().decode('ascii')

	# Pre-tokenization cleaning.
	content = re.sub('\n{2,}', '. ', content)
	content = re.sub('\.{2,}', '.', content)

	# Tokenizes the content.
	tokenized_content = tokenizer(content).lower()

	cleaned_content = cleanExtra(tokenized_content)

	# Splits on ' . ' to get individual sentences. Periods not part of
	# abbreviations have been padded with spaces around them, so that we
	# can easily identify where sentences end. 
	lines_list = cleaned_content.split(' . ')

	# Splits a sentence into a list with individual words at each index.
	# Adds a start and finish token to each sentence.
	sentences = []
	for line in lines_list:
		line = re.sub('\. ', '', line)
		line = re.sub(' \.', '', line)
		sentences.append(['<START>'] + line.split(' ') + ['<END>'])

	return sentences

def train(authors, testing_bool):
	"""
	train() takes a list of authors and trains the language models on them.
	A boolean value determines whether the user wants to train for development
	or testing. Returns nothing.

	@params: list of authors and a boolean value describing whether to train
			 for development or testing.
	@return: n/a.
	"""

	print('Training... (this may take a while)')
	for author in authors:
		# author[1] --> url.
		sentences = getData(author[1])

		if testing_bool:
			# Creates the model and pickles it.
			train_model = getNGramCounts(sentences)

			# author[0] --> name of author.
			pickle.dump(train_model, open(author[0] + 'Model.p', 'wb'))
		else:
			# Creates a test set out of 10% of the data. The remaining is our
			# development set. Both sets are pickled.
			cut_off = int(len(sentences) * 0.1)
			start = random.randint(0, len(sentences) - cut_off)
			stop = start + cut_off

			train_set = sentences[:start] + sentences[stop:]
			test_set = sentences[start:stop]
			train_model = getNGramCounts(train_set)

			# author[0] --> name of author.
			pickle.dump(train_model, open(author[0] + 'Model.p', 'wb'))
			pickle.dump(test_set, open(author[0] + 'DevTest.p', 'wb'))

def getNGramCounts(sentences_list_list):
	"""
	getNGramCounts() takes a list of sentences where each sentence has in turn
	been made into a list of individual words and reads the words/ sentences
	into a model built on trigrams. The model is returned.

	@params: sentences (list of lists).
	@return: model (dictionary).
	"""
	model = {}
	model[0] = 0
	# Loops over each word in each sentence and checks whether we have seen
	# the bigram/ trigram before (starting at the current word). Adds a count
	# or a new bigram/ trigram depending on whether we have seen it before
	# or not.
	for sentence in sentences_list_list:
		for index, word in enumerate(sentence):
			model[0] += 1
			if word in model:
				model[word][0] += 1
			else:
				model[word] = {}
				model[word][0] = 1
			# Checks whether we are too close to the end of the sentence to
			# create a bigram.
			if index + 1 < len(sentence):
				next_word = sentence[index + 1]
				if next_word in model[word]:
					model[word][next_word][0] += 1
				else:
					model[word][next_word] = {}
					model[word][next_word][0] = 1
			# Checks whether we are too close to the end of the sentence to
			# create a trigram.
			if index + 2 < len(sentence):
				next_word = sentence[index + 1]
				next_next_word = sentence[index + 2]
				if next_next_word in model[word][next_word]:
					model[word][next_word][next_next_word][0] += 1
				else:
					model[word][next_word][next_next_word] = {}
					model[word][next_word][next_next_word][0] = 1

	return model

def loadTestFile(f):
	"""
	loadTestFile() takes a file and reads it. The sentences in the file are
	tokenized, cleaned and split into indiidual words before being added to a
	list. The list of words in each sentence is returned (list of lists).

	@params: filename.
	@return: sentences contained in the file (list of lists).
	"""
	assert(os.path.isfile(f))

	content_list = []
	# Reads the file and loops over each line in it.
	test_content = open(f, 'r').read().split('\n')
	for line in test_content:
		# Tokenizes and cleans each sentence appropriately.
		tok_line = tokenizer(line)
		clean_line = cleanExtra(tok_line)
		clean_line = re.sub(' \. ', ' ', clean_line)
		clean_line = re.sub(' \.', '', clean_line)
		clean_line = re.sub('\" ', '', clean_line)
		clean_line = re.sub(' \"', '', clean_line)
		# Adds a start and end token to the sentence before adding it to the
		# list of sentences.
		content_list.append(['<START>'] + clean_line.split(' ') + ['<END>'])

	return content_list

def findProbability(sentence, model, tri_list, bi_list, uni_list):
	"""
	findProbability()

	@params: sentence (string)
			 model (dictionary)
			 tri_list (list of frequencies of frequencies for trigrams)
			 bi_list (list of frequencies of frequencies for bigrams)
			 uni_list (list of frequencies of frequencies for unigrams)
	@return: probability that the model explains the sentence.
	"""
	prob = 0
	# Loops over each word in the given sentence and finds the
	# probability of each trigram given the model.
	for i, word in enumerate(sentence):
		if i < len(sentence) - 2:
			# Checks whether the unigram has been seen.
			if word in model:
				next = sentence[i + 1]
				# Checks whether the bigram has been seen.
				if next in model[word]:
					next_next = sentence[i + 2]
					# Checks whether the trigram has been seen.
					if next_next in model[word][next]:
						# Have seen trigram
						# Gets the counts for the trigram and bigram.
						trigram = model[word][next][next_next][0]
						bigram = model[word][next][0]
						if trigram >= 6:
							# No need to smooth by GT
							prob += math.log(trigram / bigram)
						else:
							# Need to smooth by GT
							n_i_1 = tri_list[trigram + 1]
							n_i = tri_list[trigram]

							# Checks whether we need to smooth the bigram.
							if bigram < 6:
								m_i_1 = bi_list[bigram + 1]
								m_i = bi_list[bigram]
								bigram = ((bigram + 1) * (m_i_1 / m_i))

							smooth_count = ((trigram + 1) * (n_i_1 / n_i))
							prob += math.log(smooth_count / bigram)
					else:
						# Havent seen trigram
						# Have seen bigram
						# Gets the counts for the trigram and bigram.
						trigram = 0
						bigram = model[word][next][0]

						n_i = tri_list[0]
						n_i_1 = tri_list[1]

						# Checks whether we need to smooth the bigram.
						if bigram < 6:
							m_i_1 = bi_list[bigram + 1]
							m_i = bi_list[bigram]
							bigram = ((bigram + 1) * (m_i_1 / m_i))

						smooth_count = ((trigram + 1) * (n_i_1 / n_i))

						prob += math.log(smooth_count / bigram)
				else:
					# Havent seen trigram
					# Havent seen bigram
					# Have seen unigram
					# Gets the counts for the trigram and unigram.
					trigram = 0
					unigram = model[word][0] # backoff

					n_i = tri_list[0]
					n_i_1 = tri_list[1]

					# Checks whether we need to smooth the unigram.
					if unigram < 6:
						m_i_1 = uni_list[unigram + 1]
						m_i = uni_list[unigram]
						unigram = ((unigram + 1) * (m_i_1 / m_i))

					smooth_count = ((trigram + 1) * (n_i_1 / n_i))

					prob += math.log(smooth_count / unigram)
			else:
				# Havent seen trigram
				# Havent seen bigram
				# Havent seen unigram
				# Gets the counts for the trigram and possible. The possible
				# is used to penalize heavily the fact that we have not
				# seen anything like this before.
				trigram = 0
				possible = len(model) ** 3 # backoff

				n_i = tri_list[0]
				n_i_1 = tri_list[1]

				smooth_count = ((trigram + 1) * (n_i_1 / n_i))

				prob += math.log(smooth_count / possible)

	return prob

def getGTList(model, version):
	"""
	getGTList() takes a model and creates a list of frequencies of frequencies
	of n-grams. The user provides which type of n-gram (uni, bi, tri) is to be
	counted via the parameter 'version'. The Good-Turing list is returned.

	@params: model and version (uni, bi, tri).
	@return: list of frequencies of frequencies of n-grams.
	"""
	# Makes sure the user has opted to collect stats on either unigrams,
	# bigrams or trigrams.
	if version not in ['uni', 'bi','tri']:
		errorMessage()

	seen = 0
	gt_list = [0 for i in range(7)]
	power = 1

	# Loops over each unigram in the model.
	for unigram in model:
		if unigram != 0:
			# If the user wants unigrams, collect stats on that.
			if version == 'uni':
				count = model[unigram][0]
				seen += 1
				if count < 7:
					gt_list[count] += 1
			else:
				power = 2
				# Loops over each bigram in the model.
				for bigram in model[unigram]:
					if bigram != 0:
						# If the user wants bigrams, collect stats on that.
						if version == 'bi':
							count = model[unigram][bigram][0]
							seen += 1
							if count < 7:
								gt_list[count] += 1
						else:
							# The user surely must want stats on trigrams,
							# as they did not want stats on either unigrams
							# or bigrams.
							power = 3
							# Loops over each trigram in the model.
							for trigram in model[unigram][bigram]:
								if trigram != 0:
									count = model[unigram][bigram][trigram][0]
									seen += 1
									if count < 7:
										gt_list[count] += 1
	# Finds the possible number of uni-/bi-/tri-grams.
	possible = len(model) ** power
	# Finds the number of n-grams that we have not seen.
	gt_list[0] = possible - seen
	return gt_list

def testDev(authorlist):
	"""
	testDev() takes a list of authors and pickles (loads) the test sets
	created earlier for those authors. The probability of each author in
	writing each of the sentences in the test set is the ncalculated from the
	models and a 'winner' is determined for each sentence. The results are
	printed out in a format that shows how many predictions were correct.

	@params: list of authors.
	@return: n/a.
	"""
	# Compiles the test sets by pickling (loading) the test files.
	test_set = []
	model_set = {}
	for author in authorlist:
		author_test = pickle.load(open(author[0] + 'DevTest.p', 'rb'))
		for sentence in author_test:
			test_set.append([sentence, author[0], float('-inf'), None])

	# Loops over each aithor and and each sentence in the test set to find
	# the probability that the author wrote the test sentence.
	for author in authorlist:
		model = pickle.load(open(author[0] + 'Model.p', 'rb'))
		tri_list = getGTList(model, 'tri')
		bi_list = getGTList(model, 'bi')
		uni_list = getGTList(model, 'uni')
		for i, test in enumerate(test_set):
			prob = findProbability(test[0], model, tri_list, bi_list, uni_list)
			if prob > test[2]:
				test[2] = prob
				test[3] = author[0]

	print('Results on dev set:')

	# Loops over each author and each test to compile the data on how many
	# tests where correct. Prints out the results.
	for author in authorlist:
		correct = 0
		total = 0
		for test in test_set:
			if author[0] == test[1]:
				if test[1] == test[3]:
					correct += 1
				total += 1

		print(author[0] + ': ' + str(correct) + '/' + str(total) + ' correct')

def developModel(authorlist):
	"""
	developModel()

	@params: list of authors (filename).
	@return: n/a.
	"""
	# Gets the information on the authors from the file.
	authors = openAuthorlist(authorlist)
	# Trains the model on the authors.
	train(authors, False)
	# Tests the models that were built.
	testDev(authors)

def testModel(authorlist, test_file):
	"""
	testModel() takes a list of authors (filename), opens it and then trains
	a n-gram model on each author. After training is completed the models
	are tested against the test file provided and the results are printed out.

	@params: list of authors (filename) and name of test file.
	@return: n/a.
	"""
	# Gets the information on the authors from the file.
	authors = openAuthorlist(authorlist)
	# Trains the model on the authors.
	train(authors, True)
	# Loads the content of the test file.
	test_set = loadTestFile(test_file)

	result = [[float('-inf'), None] for i in range(len(test_set))]
	# Loops over each author and each sentence in the test file and compares
	# the probabilities that each author wrote the sentence.
	for author in authors:
		model = pickle.load(open(author[0] + 'Model.p', 'rb'))
		tri_list = getGTList(model, 'tri')
		bi_list = getGTList(model, 'bi')
		uni_list = getGTList(model, 'uni')
		for i, sent in enumerate(test_set):
			prob = findProbability(sent, model, tri_list, bi_list, uni_list)
			if prob >= result[i][0]:
				result[i] = [prob, author[0]]

	# Prints out the results.
	for item in result:
		print(item[1])

def main():
	# Legal command line formats:
	# $ classifier.py -dev authorlist
	# $ classifier.py -test authorlist testset.txt

	if len(sys.argv) == 3:
		if sys.argv[1] == '-dev':
			# Checks whether the authorlist file exists in the current
			# directory.
			if os.path.isfile(sys.argv[2]):
				developModel(sys.argv[2])
			else:
				errorMessage()
		else:
			errorMessage()
	elif len(sys.argv) == 4:
		if sys.argv[1] == '-test':
			# Checks whether the authorlist file exists in the current
			# directory.
			if os.path.isfile(sys.argv[2]) and os.path.isfile(sys.argv[3]):
				testModel(sys.argv[2], sys.argv[3])
			else:
				errorMessage()
		else:
			errorMessage()
	else:
		errorMessage()


if __name__ == '__main__':
	main()