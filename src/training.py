import csv
import globals as g
import loadData
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(g.device)
    return loss, nTotal.item()

def train(inputVariable, lengths, targetVariable, mask, maxTargetLen, encoder, decoder,
          encoderOptimizer, decoderOptimizer, batchSize, clip, teacherForcingRatio):

    # Zero gradients
    encoderOptimizer.zero_grad()
    decoderOptimizer.zero_grad()

    # Set device options
    inputVariable = inputVariable.to(g.device)
    targetVariable = targetVariable.to(g.device)
    mask = mask.to(g.device)
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    printLosses = []
    n_totals = 0

    # Forward pass through encoder
    encoderOutputs, encoderHidden = encoder(inputVariable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoderInput = torch.LongTensor([[g.SOS_token for _ in range(batchSize)]])
    decoderInput = decoderInput.to(g.device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoderHidden = {'h': encoderHidden['h'][:decoder.numLayers]}
    if 'c' in encoderHidden.keys(): decoderHidden['c'] = encoderHidden['c'][:decoder.numLayers]

    # Determine if we are using teacher forcing this iteration
    useTeacherForcing = True if random.random() < teacherForcingRatio else False

    # Forward batch of sequences through decoder one time step at a time
    if useTeacherForcing:
        for t in range(maxTargetLen):
            decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
            # Teacher forcing: next input is current target
            decoderInput = targetVariable[t].view(1, -1)

            # Calculate and accumulate loss
            maskLoss, nTotal = maskNLLLoss(decoderOutput, targetVariable[t], mask[t])
            loss += maskLoss
            printLosses.append(maskLoss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(maxTargetLen):
            decoderOutput, decoderHidden = decoder(decoderInput, decoderHidden, encoderOutputs)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoderOutput.topk(1)
            decoderInput = torch.LongTensor([[topi[i][0] for i in range(batchSize)]])
            decoderInput = decoderInput.to(g.device)

            # Calculate and accumulate loss
            maskLoss, nTotal = maskNLLLoss(decoderOutput, targetVariable[t], mask[t])
            loss += maskLoss
            printLosses.append(maskLoss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoderOptimizer.step()
    decoderOptimizer.step()

    return sum(printLosses) / n_totals

def trainIters(modelName, voc, pairs, encoder, decoder, encoderOptimizer, decoderOptimizer, 
               embedding, hiddenSize, encoderNumLayers, decoderNumLayers, saveDir, numIterations, 
               batchSize, printEvery, saveEvery, clip, corpusName, doLoadModelConfigFromFile, checkpoint,
               teacherForcingRatio):

    # Load batches for each iteration
    training_batches = [loadData.batch2TrainData(voc, [random.choice(pairs) for _ in range(batchSize)]) for _ in range(numIterations)]

    # Initializations
    print('Initializing ...')
    startIteration = 1
    printLoss = 0
    lossOverTime = []
    if doLoadModelConfigFromFile:
        startIteration = checkpoint['iteration'] + 1
    directory = os.path.join(saveDir, modelName, corpusName, '{}-{}_{}'.format(encoderNumLayers, decoderNumLayers, hiddenSize))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Training loop
    print("Training...")
    for i, iteration in tqdm(enumerate(range(startIteration, numIterations + 1))):
        # Grab the next training batch.
        trainingBatch = training_batches[iteration - 1]
        inputVariable, lengths, targetVariable, mask, maxTargetLen = trainingBatch

        # Run a training iteration with batch
        loss = train(inputVariable, lengths, targetVariable, mask, maxTargetLen, encoder, decoder, 
                     encoderOptimizer, decoderOptimizer, batchSize, clip, teacherForcingRatio)
        printLoss += loss
        lossOverTime.append([i, loss])

        # Print progress
        if iteration % printEvery == 0:
            printLossAvg = printLoss / printEvery
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / numIterations * 100, printLossAvg))
            printLoss = 0

        # Save checkpoint
        if (iteration % saveEvery == 0):
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoderOptimizer.state_dict(),
                'de_opt': decoderOptimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
    
    # Log loss over time to file.
    lossFile = os.path.join(directory, 'loss.csv')
    with open(lossFile, 'w', newline='') as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerows(lossOverTime)
    csvFile.close()

def callAndResponse(searcher, voc, sentence, maxLength):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [loadData.indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(g.device)
    lengths = lengths.to("cpu") # We're going to pack the input sequence, and Pytorch needs lengths to be on the cpu for that operation :(
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, maxLength)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

def evaluate(testCall, testResponse, searcher, voc, maxLength, saveDir, modelName, corpusName, encoderNumLayers, decoderNumLayers, hiddenSize):
    # Initialize values
    directory = os.path.join(saveDir, modelName, corpusName, '{}-{}_{}'.format(encoderNumLayers, decoderNumLayers, hiddenSize))
    if not os.path.exists(directory):
        os.makedirs(directory)
    meteorScores = []
    bleuScores = []

    # Read through the call and ground truth response files.
    # For each call, ask the bot to generate a response.
    # Evaluate the bot response against the ground truth response for METEOR and BLEU scores.
    with open(testCall, 'r', encoding='utf-8') as testCallFile:
        with open(testResponse, 'r', encoding='utf-8') as testResponseFile:
            for call in testCallFile.readlines():
                try:
                    call = loadData.normalizeString(call)
                    gtResp = loadData.normalizeString(testResponseFile.readline())
                    botResp = " ".join(callAndResponse(searcher, voc, call, maxLength))

                    meteorScores += [meteor_score([gtResp], botResp)]
                    ref = [[word_tokenize(gtResp.lower())]]
                    hyp = [word_tokenize(botResp.lower())]
                    bleuScores += [corpus_bleu(ref, hyp, weights=(.5,.5))]
                except KeyError:
                    pass # move on quietly
    
    # Log the scores.
    print('METEOR: {}'.format(np.mean(meteorScores)))
    print('BLEU: {}'.format(np.mean(bleuScores)))
    metrics = os.path.join(directory, 'metrics.txt')
    with open(metrics, 'w') as metricsFile:
        metricsFile.write('METEOR: {}'.format(np.mean(meteorScores)))
        metricsFile.write("\n")
        metricsFile.write('BLEU: {}'.format(np.mean(bleuScores)))
    metricsFile.close()

def interact(searcher, voc, maxLength):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = loadData.normalizeString(input_sentence)
            # Evaluate sentence
            output_words = callAndResponse(searcher, voc, input_sentence, maxLength)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")