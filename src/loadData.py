import codecs
import csv
import itertools
import os
import re
import torch
import unicodedata
import globals as g

def prepareData(corpusName, movieLinesName, movieConversationsName, formattedLinesName, doFormatMovieLines, minWordCountForTrim, maxLength, toTestEvery=1000):
    corpusDir = os.path.join("data", corpusName)
    movieLines = os.path.join(corpusDir, movieLinesName)
    movieConversations = os.path.join(corpusDir, movieConversationsName)
    dataFile = os.path.join(corpusDir, formattedLinesName)

    # Load raw corpus data and write it to a formatted output.
    if doFormatMovieLines:
        delimiter = str(codecs.decode("\t", "unicode_escape"))
        formatMovieLines(corpusDir, movieLines, movieConversations, dataFile, delimiter, toTestEvery)

    # Read formatted data into memory.
    voc, pairs = loadPrepareData(corpusName, dataFile, maxLength)
    pairs = trimRareWords(voc, pairs, minWordCountForTrim)
    return voc, pairs


##########################################
# Read the raw data and write it 
# to a formatted file.
##########################################

def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines

def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
            convObj['lines'] = []
            for lineId in lineIds:
                convObj['lines'].append(lines[lineId])
            conversations.append(convObj)
    return conversations

def extractSentencePairs(conversations):
    qaPairs = []
    for conversation in conversations:
        for i in range(len(conversation['lines'])-1):
            inputLine = conversation['lines'][i]['text'].strip()
            targetLine = conversation['lines'][i+1]['text'].strip()
            if inputLine and targetLine:
                qaPairs.append([inputLine, targetLine])
    return qaPairs

def formatMovieLines(dataDir, linesFile, conversationsFile, dataFile, delimiter, toTestEvery=1000):
    # Load the raw data.
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
    print("Loading lines...")
    lines = loadLines(linesFile, MOVIE_LINES_FIELDS)
    print("Loading conversations...")
    conversations = loadConversations(conversationsFile, lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write a newly formatted output file.
    print("Writing formatted data file....")
    i = 1
    testCall = os.path.join(dataDir, 'test_call.txt')
    testResponse = os.path.join(dataDir, 'test_resp.txt')
    with open(testCall, 'w', encoding='utf-8') as  testCallFile:
        with open(testResponse, 'w', encoding='utf-8') as testResponseFile:
            with open(dataFile, 'w', encoding='utf-8') as outputFile:
                writer = csv.writer(outputFile, delimiter=delimiter, lineterminator="\n")
                for pair in extractSentencePairs(conversations):
                    if (i % toTestEvery) == 0:
                        testCallFile.write(pair[0]+"\n")
                        testResponseFile.write(pair[1]+"\n")
                    else:
                        writer.writerow(pair)
                    i += 1
    print("Data load and format complete!")


##########################################
# Assemble the vocabulary.
##########################################

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(dataFile, corpusName):
    print("Reading lines...")
    lines = open(dataFile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpusName)
    return voc, pairs

def filterPair(p, maxLength):
    return len(p[0].split(' ')) < maxLength and len(p[1].split(' ')) < maxLength

def filterPairs(pairs, maxLength):
    return [pair for pair in pairs if filterPair(pair, maxLength)]

def loadPrepareData(corpusName, datafile, maxLength):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpusName)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, maxLength)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.numWords)
    return voc, pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


##########################################
# Convert data in memory to tensors.
##########################################

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [g.EOS_token]

def zeroPadding(l, fillvalue=g.PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=g.PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == g.PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


##########################################
# Corpus Vocabulary
##########################################

class Voc:

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {g.PAD_token: "PAD", g.SOS_token: "SOS", g.EOS_token: "EOS", g.UNK_token: "UNK"}
        self.numWords = 4

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.numWords
            self.word2count[word] = 1
            self.index2word[self.numWords] = word
            self.numWords += 1
        else:
            self.word2count[word] += 1
    
    def trim(self, minCount):
        if self.trimmed:
            return
        self.trimmed = True

        keepWords = []
        for k, v in self.word2count.items():
            if v >= minCount:
                keepWords.append(k)

        self.word2index = {}
        self.word2count = {}
        self.index2word = {g.PAD_token: "PAD", g.SOS_token: "SOS", g.EOS_token: "EOS", g.UNK_token: "UNK"}
        self.numWords = 4
        for word in keepWords:
            self.addWord(word)