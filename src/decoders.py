import torch
import torch.nn as nn
import torch.nn.functional as F

def buildDecoder(attentionMethod, embedding, hiddenSize, outputSize, numLayers=1, dropout=0, decoderMethod="gru"):
    decoder = None
    if decoderMethod == "gru":
        decoder = GruAttnDecoder(attentionMethod, embedding, hiddenSize, outputSize, numLayers, dropout)
    elif decoderMethod == "lstm":
        decoder = LstmAttnDecoder(attentionMethod, embedding, hiddenSize, outputSize, numLayers, dropout)
    else:
        raise ValueError("'decoderMethod' must be one of: gru, lstm")
    return decoder


class GruAttnDecoder(nn.Module):
    def __init__(self, attentionMethod, embedding, hiddenSize, outputSize, numLayers=1, dropout=0.1):
        super(GruAttnDecoder, self).__init__()

        # Keep for reference
        self.attentionMethod = attentionMethod
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hiddenSize, hiddenSize, numLayers, dropout=(0 if numLayers == 1 else dropout))
        self.concat = nn.Linear(hiddenSize * 2, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.attn = Attn(attentionMethod, hiddenSize)

    def forward(self, inputStep, lastHidden, encoderOutputs):
        # Embeddings
        embedded = self.embedding(inputStep)
        embedded = self.embedding_dropout(embedded)

        # GRU oputput and hidden state
        rnn_output, h = self.gru(embedded, lastHidden['h'])
        hidden = {'h': h}

        # Attention!
        attn_weights = self.attn(rnn_output, encoderOutputs)
        context = attn_weights.bmm(encoderOutputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict the output using softmax and return.
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden


class LstmAttnDecoder(nn.Module):
    def __init__(self, attentionMethod, embedding, hiddenSize, outputSize, numLayers=1, dropout=0.1):
        super(LstmAttnDecoder, self).__init__()

        # Keep for reference
        self.attentionMethod = attentionMethod
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.numLayers = numLayers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hiddenSize, hiddenSize, numLayers, dropout=(0 if numLayers == 1 else dropout))
        self.concat = nn.Linear(hiddenSize * 2, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.attn = Attn(attentionMethod, hiddenSize)

    def forward(self, inputStep, lastHidden, encoderOutputs):
        # Embeddings
        embedded = self.embedding(inputStep)
        embedded = self.embedding_dropout(embedded)

        # LSTM output and state
        rnnOutput, (h, c) = self.lstm(embedded, (lastHidden['h'], lastHidden['c']))
        hidden = {'h': h, 'c': c}

        # Attention
        attnWeights = self.attn(rnnOutput, encoderOutputs)
        context = attnWeights.bmm(encoderOutputs.transpose(0, 1))
        rnnOutput = rnnOutput.squeeze(0)
        context = context.squeeze(1)
        concatInput = torch.cat((rnnOutput, context), 1)
        concatOutput = torch.tanh(self.concat(concatInput))

        # Predict the outputs using softmax and return
        output = self.out(concatOutput)
        output = F.softmax(output, dim=1)
        return output, hidden


class Attn(nn.Module):
    def __init__(self, method, hiddenSize):
        super(Attn, self).__init__()
        self.hidden_size = hiddenSize
        self.method = method

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hiddenSize)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hiddenSize)
            self.v = nn.Parameter(torch.FloatTensor(hiddenSize))
        elif method != "dot":
            raise ValueError("'method' must be one of: general, concat, dot")

    def dotScore(self, hidden, encoderOutputs):
        return torch.sum(hidden * encoderOutputs, dim=2)

    def generalScore(self, hidden, encoderOutputs):
        energy = self.attn(encoderOutputs)
        return torch.sum(hidden * energy, dim=2)

    def concatScore(self, hidden, encoderOutputs):
        energy = self.attn(torch.cat((hidden.expand(encoderOutputs.size(0), -1, -1), encoderOutputs), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoderOutputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.generalScore(hidden, encoderOutputs)
        elif self.method == 'concat':
            attn_energies = self.concatScore(hidden, encoderOutputs)
        elif self.method == 'dot':
            attn_energies = self.dotScore(hidden, encoderOutputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)