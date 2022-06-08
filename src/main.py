import decoders
import encoders
import globals as g
import loadData
import modelconfigs
import os
import outputdecoders
import torch
from torch import optim
import torch.nn as nn
import training

#######################################
#######################################
# Application controls.
modelConfig = modelconfigs.Lstm_Dot_TopP_2_2_1500_ML20
doFormatMovieLines = True
doTraining = True
doEvaluation = True
doLoadModelConfigFromFile = False


def model(modelConfig, doFormatMovieLines, doTraining, doEvaluation, doLoadModelConfigFromFile, doStartModel):
    #######################################
    # Model & Training Config

    # Data load
    minWordCountForTrim = modelConfig['minWordCountForTrim']
    maxLength = modelConfig['maxLength']
    sendPairToTestEveryNIterations = modelConfig['sendPairToTestEveryNIterations']

    # Model config
    modelName = modelConfig['modelName']
    encoderDecoderMethod = modelConfig['encoderDecoderMethod']
    attnMethod = modelConfig['attnMethod']
    searchMethod = modelConfig['searchMethod']
    topp = modelConfig['topp']
    topk = modelConfig['topk']
    hiddenSize = modelConfig['hiddenSize']
    encoderNumLayers = modelConfig['encoderNumLayers']
    decoderNumLayers = modelConfig['decoderNumLayers']
    dropout = modelConfig['dropout']
    batchSize = modelConfig['batchSize']

    # Training config
    clip = modelConfig['clip']
    teacherForcingRatio = modelConfig['teacherForcingRatio']
    learningRate = modelConfig['learningRate']
    decoderLearningRatio = modelConfig['decoderLearningRatio']
    numIterations = modelConfig['numIterations']

    # Corpus Info
    corpusName = "cornell movie-dialogs corpus"
    movieLinesName = "movie_lines.txt"
    movieConversationsName = "movie_conversations.txt"
    formattedLinesName = "formatted_movie_lines.txt"

    # Checkpoint save config
    printEvery = 500
    saveEvery = numIterations
    saveDir = "save"
    checkpointIter = numIterations
    loadFilename = os.path.join(saveDir, modelName, corpusName, '{}-{}_{}'.format(encoderNumLayers, decoderNumLayers, hiddenSize), '{}_checkpoint.tar'.format(checkpointIter))

    #######################################
    # Load training data
    voc = loadData.Voc(corpusName)
    if doTraining:
        voc, pairs = loadData.prepareData(corpusName, movieLinesName, movieConversationsName, formattedLinesName, doFormatMovieLines, minWordCountForTrim, maxLength, sendPairToTestEveryNIterations)

    #######################################
    # Load model if a loadFilename is provided
    checkpoint = None
    if doLoadModelConfigFromFile:
        print("Restoring model and vocab from file....")
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
        print("Restoration complete!")

    #######################################
    # Build the model
    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.numWords, hiddenSize)
    if doLoadModelConfigFromFile:
        embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = encoders.buildEncoder(hiddenSize, embedding, encoderNumLayers, dropout, encoderDecoderMethod)
    decoder = decoders.buildDecoder(attnMethod, embedding, hiddenSize, voc.numWords, decoderNumLayers, dropout, encoderDecoderMethod)
    if doLoadModelConfigFromFile:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(g.device)
    decoder = decoder.to(g.device)
    print('Models built and ready to go!')

    #######################################
    # Train the model, if required.
    if doTraining:
        # Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()

        # Initialize optimizers
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learningRate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learningRate * decoderLearningRatio)
        if doLoadModelConfigFromFile:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        # If you have cuda, configure cuda to call
        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # Run training iterations
        print("Starting Training!")
        training.trainIters(modelName, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                            embedding, hiddenSize, encoderNumLayers, decoderNumLayers, saveDir, numIterations, 
                            batchSize, printEvery, saveEvery, clip, corpusName, doLoadModelConfigFromFile, checkpoint,
                            teacherForcingRatio)

    #######################################
    # Build the output decoder.
    searcher = outputdecoders.buildSearchDecoder(encoder, decoder, searchMethod, topp, topk)
    searcher.to(g.device)

    #######################################
    # Evaluate the model, spitting out BLEU and METEOR scores.
    if doEvaluation:
        # Load up the training call and response text.
        # The evaluation will query the model with a call.
        # The model response will be compared to the test responses, which shall be the ground truth.
        encoder.eval()
        decoder.eval()
        dataDir = os.path.join('data', corpusName)
        testCall = os.path.join(dataDir, 'test_call.txt')
        testResp = os.path.join(dataDir, 'test_resp.txt')
        training.evaluate(testCall, testResp, searcher, voc, maxLength, saveDir, modelName, corpusName, encoderNumLayers, decoderNumLayers, hiddenSize)

    #######################################
    # Interact with the model.
    if doStartModel:
        # Set dropout layers to eval mode
        # Then begin chatting!
        encoder.eval()
        decoder.eval()
        training.interact(searcher, voc, maxLength)

    # Return the model.
    return searcher, voc, maxLength