# "EVERYTHING I KNOW, I LEARNED FROM THE MOVIES"
Basic RNN Chatbot, built on encoder-decoder architecture, trained on movie dialogue.

!!! If you want to run this out-of-the-box !!!
`python main.py`

## Table of Contents
1. Build Environment
2. Python Requirements
3. Model Execution
4. Code Layout

## 1. Build Environment
Lang: Python 3.8.3 (Conda 4.8.3)
IDE:  Visual Studio Code (Extensions: Python; Pylance)
OS:   Win10
CUDA Drivers: 11.1.105

## 2. Python Requirements
~Python Version~
- Python 3.8.3
- Conda 4.8.3

~ Installed Libs~
- nltk
- numpy
- pytorch
- tqdm

## 3. Model Execution ##
~TLDR~
`python main.py`
Running this as-is will:
- Format the raw training data
- Train a model using GRU cells (2 layers x 500 cells on both the encoder and decoder), dot-product attention, and TopP=.4 output decoder
- Evaluate the model on holdout test data (METEOR, BLEU)
- Make the chatbot available to use (talk to it at the prompt, enter "q" to quit)

~The Long Version~
Python file "main.py" controls the flow of the entire application. At the top of the file are two clearly indicated sections that are intended to be manually edited, in order to dictate how the chatbot should execute. We'll go over each of those sections here.

### Application controls.
Everything needed to control the application is contained here:
- modelConfig: Assigns the model configuration to use. Prebuilt model configurations are stored in modelconfigs.py.
- doFormatMovieLines: If True, the raw data files will be read and formatted training and test files will be written. This is required the first time you run the application, after which this may be set to False. However, if the 'maxLength' property on a model configuration is altered, this will need to be set to True again to-rewrite the formatted data (maxLength sets the maximum length allowed for training sentences in tokens). If True, a new model will be trained, according to the model configuration defined above. If this is False, property 'doLoadModelConfigFromFile' must be set to True and an existing saved-to-file model must exist (models will be saved automatically).
- doEvaluation: If True, the model trained and/or loaded from file will be evaluated on hold-out test data. METEOR and BLEU scores will be written to file.
- doLoadModelConfigFromFile: If True, a previously saved-to-file model will be loaded. Saved models are fully trained and ready for use. If this is False, then 'doTraining' must be True, so that a new model may be built.

### Model & Training Config
More fine-tuned controls are here. Model configurations are intended to be loaded using property 'modelConfig'. However, the following properties may be manually edited, overriding model config. The explanations given here apply to modelconfigs.py as well.
- minWordCountForTrim: Num instances of a word in order to be included in the vocab.
- maxLength: Maximum sentence length for calls and responses in the training data. Sentences longer will be ignored.
- sendPairToTestEveryNIterations: Num iterations between sending a dialogue line to test rather than training.
- modelName: Named identifier for the model configuration.
- encoderDecoderMethod: RNN cell type. Applies to both encoder and decoder blocks. Must be one of: gru, lstm
- attnMethod: Attention scoring algorithm: Must be one of: dot, general, concat
- searchMethod: Output decoder algorithm. Must be one of: greedy, topp, topk
- topp: Probability cap for TopP. May be 0 if TopP is not the searchMethod.
- topk: Num softmax options to consider for output with TopK.
- hiddenSize: Num neurons in a layer. Applies to both encoder and decoder blocks.
- encoderNumLayers: Num encoder RNN layers.
- decoderNumLayers: Num decoder RNN layers.
- dropout: Dropout ratio. Applies to both encoder and decoder blocks.
- batchSize: Num training samples to include in a batch.
- clip: Gradient clipping ratio.
- teacherForcingRatio: During training, probability of teacher forcing input to decoder at time T.
- learningRate: Sets the Adam-optimizer learning rate (use of Adam over other optimizers is hard-coded, maybe that should also be a hyperparam).
- decoderLearningRatio: Modifies learning rate for decoder optimizer.
- numIterations: Num training iterations.

The following properties define the corpus to load and data save behaviour. Left alone, the movie dialogue corpus will be loaded and models will be saved only when training has completed.
- corpusName: Identifier given to the corpus.
- movieLinesName: Movie lines file.
- movieConversationsName: Movie conversations file.
- formattedLinesName: File to write formatted training data to.
- printEvery: Num training iterations between printing average loss data.
- saveEvery: Num training iterations between saving current model to file.
- saveDir: Directory to save models to.
- checkpointIter: If loading a model from file, specifies the checkpoint (num training iterations) from which to load.
- loadFilename: If loading a model from file, specifies the model file name to load.

## 4. Code Layout
main.py
Defines the main logic loop of the application. All model, training, and eval configuration is defined here.

modelconfigs.py
Stores pre-built model configurations. The models defined here are references in main.py, 'modelConfig' (line 16).

globals.py
Defines some global variables.
NOTE: Control over CUDA vs CPU for Pytorch is defined here and should be overridden here, if desired.

training.py
All model training logic is controlled here. Additionally, test evaluation and control over trained model use is here as well.

loadData.py
Reads and formats raw training data. Loads formatted training data into tensors for training. Defines the corpus vocabulary.

encoders.py
RNN neurons for the encoder block are implemented here: LSTM and GRU. Also provides a factory method for building the encoder.

decoders.py
RNN neurons for the decoder implementation. Provides both LSTM and GRU implementations, Attention implementation, and a factory method for building decoders.

outputdecoders.py
Algorithm implementations for decoding RNN output to words. Contains implementations for Greedy, TopP, and TopK.

/data/*
All raw corpus data. Formatted corpus data will be written here, for both training and test evaluation.