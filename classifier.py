'''
classifier.py
Frederik Roenn Stensaeth
10.05.15

A Python program that, when given a snippet of text, will determine the most
likely corrcet author (of the given authors).

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

def tokenizer(sentence):
	"""
	tokenizer() takes a sentence (string) and tokenizes it according to the
	algorithm for tokenization provided by Jurafsky and Martin in
	'Speech and Language Processing' (2nd) on page 71.

	@params: senetence to be tokenized.
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

def makeLanguageModel(lines):
	"""
	makeLanguageModel() Xx

	@params: Xx
	@return: Xx
	"""

	# Creates a 2D table that can fit ten words. Initialize all values to be
	# None.
	bigram_table = [[None for i in range(1)] for j in range(1)]

	for line in lines:
		first = None
		second = None
		for word in line.split(' '):
			# Check if we have already seen the word or not. Add it to the
			# table if we havent.
			# Record the bigram count.
			if word in bigram_table[0]:
				# The word has been seen before.
				index_second = bigram_table[0].index(word)
				second = word
				
				# Increment the count of the bigram.
				if first != None:
					index_first = bigram_table[0].index(first)
					bigram_table[index_first][index_second] += 1
					if bigram_table[index_first][index_second] > 15:
						print(bigram_table[index_first])

				# CODE

				first = second
			else:
				# The word has not been seen before.
				# Creates a new row filled with 0s. We have not seen the word
				# before, so it has not been spotted in any bigrams.
				new_row = [0 for i in range(len(bigram_table) + 1)]
				# Sets first item in the row to be the word.
				new_row[0] = word
				# Add a 0 to the row of every other word to reflect that no
				# bigrams containing this word has been seen.
				for row in bigram_table:
					row.append(0)
				# Corrects the first row to display the word instead of a 0.
				bigram_table[0][-1] = word
				# Adds the new row to the table.
				bigram_table.append(new_row)


				index_second = bigram_table[0].index(word)
				second = word
				
				# Increment the count of the bigram.
				if first != None:
					index_first = bigram_table[0].index(first)
					bigram_table[index_first][index_second] += 1
					if bigram_table[index_first][index_second] > 15:
						print(bigram_table[index_first])

				# CODE

				first = second			

	return bigram_table

def openUrlAndClean(url):
	"""
	openUrlAndClean() takes a url and opens/ reads it. The content of the url
	is cleaned before indivudal sentences are made into a list. This list is
	returned.

	@params: Url (string).
	@return: List of indiviudal sentences (list of strings).
	"""
	# Opens the url and get the content.
	content = urllib.urlopen(url).read()
	# Pre-tokenization cleaning.
	content = re.sub('\n{2,}', '. ', content)
	content = re.sub('\.{2,}', '.', content)
	# Tokenizes the content.
	tokenized_content = tokenizer(content)

	# Cleans the content, meaning: remove non-letter, number and period
	# characters. Removes the period at the end of the file, as we later
	# want to split on periods to get individual sentences.
	cleaned_content = re.sub(' [^A-Za-z0-9.] ', ' ', tokenized_content)
	cleaned_content = re.sub('(^ )|( $)', '', cleaned_content)
	cleaned_content = re.sub(' .$', '', cleaned_content)

	# Splits on ' . ' to get individual sentences. Periods not part of
	# abbreviations have been padded with spaces around them, so that we
	# can easily identify where sentences end. 
	lines_content = cleaned_content.split(' . ')

	return lines_content

def testStuff():
	content = urllib.urlopen('http://www.cs.carleton.edu/faculty/aexley/authors/austen.txt').read()
	content = re.sub('\n{2,}', '. ', content)
	content = re.sub('\.{2,}', '.', content)
	content = classifier.tokenizer(content)
	cleaned_content = re.sub(' [^A-Za-z0-9.] ', ' ', content)
	cleaned_content = re.sub('(^ )|( $)', '', cleaned_content)
	cleaned_content = re.sub(' .$', '', cleaned_content)
	lines_content = cleaned_content.split(' . ')
	table = makeLanguageModel(lines_content)

	print(table)

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
		authors.append(line.split())

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
	sys.exit()

def main():
	# Legal command line formats:
	# $ classifier.py -dev authorlist
	# $ classifier.py -test authorlist testset.txt

	# if len(sys.argv) == 3:
	# 	if sys.argv[1] == '-dev':
	# 		# Checks whether the authorlist file exists in the current
	# 		# directory.
	# 		if os.path.isfile(sys.argv[2]):
	# 			print() # CODE
	# 		else:
	# 			errorMessage()
	# 	else:
	# 		errorMessage()
	# elif len(sys.argv) == 4:
	# 	if sys.argv[1] == '-test':
	# 		# Checks whether the authorlist file exists in the current
	# 		# directory.
	# 		if os.path.isfile(sys.argv[2]):
	# 			print() # CODE
	# 		else:
	# 			errorMessage()
	# 	else:
	# 		errorMessage()
	# else:
	# 	errorMessage()

	content = str(urllib.request.urlopen('http://www.cs.carleton.edu/faculty/aexley/authors/austen.txt').read())
	print(content)
	content = re.sub('\n{2,}', '. ', content)
	content = re.sub('\.{2,}', '.', content)
	content = tokenizer(content)
	cleaned_content = re.sub(' [^A-Za-z0-9.] ', ' ', content)
	cleaned_content = re.sub('(^ )|( $)', '', cleaned_content)
	cleaned_content = re.sub(' .$', '', cleaned_content)
	lines_content = cleaned_content.split(' . ')
	print('entered')
	table = makeLanguageModel(lines_content)

	print(table)




if __name__ == '__main__':
	main()