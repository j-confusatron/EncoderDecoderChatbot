import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):

    def __init__(self, vocSize, hiddenSize, encLayers, encDropout):
        super(Seq2Seq, self).__init__()
        self.encoder = LstmEncoder(vocSize, hiddenSize, encLayers, encDropout)

class LstmEncoder(nn.Module):

    def __init__(self, vocSize, hiddenSize, numLayers=1, dropout=0):
        super(LstmEncoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding(vocSize, hiddenSize)
        self.lstm = nn.LSTM(hiddenSize, hiddenSize, numLayers, dropout=(0 if numLayers == 1 else dropout), bidirectional=True)
    
    def forward(self, inputSeq, inputMask):
        embedded = self.embedding(inputSeq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputLens)
        outputs, h = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hiddenSize] + outputs[:, :, self.hiddenSize:]
        return outputs, h

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