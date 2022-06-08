Gru_Dot_Greedy_2_2_500 = {
    'minWordCountForTrim': 3,
    'maxLength': 10,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'gru_dot_greedy',
    'encoderDecoderMethod': "gru", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'greedy', #topp, topk, greedy
    'topp': .4,
    'topk': 0,
    'hiddenSize': 500,
    'encoderNumLayers': 2,
    'decoderNumLayers': 2,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 4000
}

Gru_Dot_TopP_2_2_500 = {
    'minWordCountForTrim': 3,
    'maxLength': 10,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'gru_dot_topp',
    'encoderDecoderMethod': "gru", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'topp', #topp, topk, greedy
    'topp': .4,
    'topk': 0,
    'hiddenSize': 500,
    'encoderNumLayers': 2,
    'decoderNumLayers': 2,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 4000
}

Gru_Dot_TopK_2_2_500 = {
    'minWordCountForTrim': 3,
    'maxLength': 10,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'gru_dot_topk',
    'encoderDecoderMethod': "gru", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'topk', #topp, topk, greedy
    'topp': 0,
    'topk': 32,
    'hiddenSize': 500,
    'encoderNumLayers': 2,
    'decoderNumLayers': 2,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 4000
}

Lstm_Dot_Greedy_2_2_500 = {
    'minWordCountForTrim': 3,
    'maxLength': 10,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'lstm_dot_greedy',
    'encoderDecoderMethod': "lstm", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'greedy', #topp, topk, greedy
    'topp': .4,
    'topk': 0,
    'hiddenSize': 500,
    'encoderNumLayers': 2,
    'decoderNumLayers': 2,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 4000
}

Lstm_Dot_TopP_2_2_500 = {
    'minWordCountForTrim': 3,
    'maxLength': 10,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'lstm_dot_topp',
    'encoderDecoderMethod': "lstm", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'topp', #topp, topk, greedy
    'topp': .4,
    'topk': 0,
    'hiddenSize': 500,
    'encoderNumLayers': 2,
    'decoderNumLayers': 2,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 4000
}

Lstm_Dot_TopP_4_4_1000_ML10 = {
    'minWordCountForTrim': 3,
    'maxLength': 10,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'lstm_dot_topp_ml10',
    'encoderDecoderMethod': "lstm", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'topp', #topp, topk, greedy
    'topp': .4,
    'topk': 0,
    'hiddenSize': 1000,
    'encoderNumLayers': 4,
    'decoderNumLayers': 4,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 16000
}

Lstm_Dot_TopP_2_2_1000_ML20 = {
    'minWordCountForTrim': 3,
    'maxLength': 20,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'lstm_dot_topp_ml20',
    'encoderDecoderMethod': "lstm", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'topp', #topp, topk, greedy
    'topp': .4,
    'topk': 0,
    'hiddenSize': 1000,
    'encoderNumLayers': 2,
    'decoderNumLayers': 2,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 10000
}

Lstm_Dot_TopP_2_2_1000 = {
    'minWordCountForTrim': 3,
    'maxLength': 10,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'lstm_dot_topp',
    'encoderDecoderMethod': "lstm", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'topp', #topp, topk, greedy
    'topp': .4,
    'topk': 0,
    'hiddenSize': 1000,
    'encoderNumLayers': 2,
    'decoderNumLayers': 2,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 5000
}

Lstm_Dot_TopP_2_2_1500_ML20 = {
    'minWordCountForTrim': 3,
    'maxLength': 20,
    'sendPairToTestEveryNIterations': 1000,
    'modelName': 'lstm_dot_topp_ml20',
    'encoderDecoderMethod': "lstm", # gru, lstm
    'attnMethod': 'dot', # dot, general, concat
    'searchMethod': 'topp', #topp, topk, greedy
    'topp': .3,
    'topk': 0,
    'hiddenSize': 1500,
    'encoderNumLayers': 2,
    'decoderNumLayers': 2,
    'dropout': 0.1,
    'batchSize': 64,
    'clip': 50.0,
    'teacherForcingRatio': 0.95,
    'learningRate': 0.0001,
    'decoderLearningRatio': 5.0,
    'numIterations': 6000
}