import torch.nn as nn

def buildEncoder(hiddenSize, embedding, numLayers=1, dropout=0, encoderMethod="gru"):
    encoder = None
    if encoderMethod == "gru":
        encoder = GruEncoder(hiddenSize, embedding, numLayers, dropout)
    elif encoderMethod == "lstm":
        encoder = LstmEncoder(hiddenSize, embedding, numLayers, dropout)
    else:
        raise ValueError("'encoderMethod' must be one of: gru, lstm")
    return encoder


class GruEncoder(nn.Module):

    def __init__(self, hiddenSize, embedding, numLayers=1, dropout=0):
        super(GruEncoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.embedding = embedding
        self.method = "gru"
        self.gru = nn.GRU(hiddenSize, hiddenSize, numLayers, dropout=(0 if numLayers == 1 else dropout), bidirectional=True)
    
    def forward(self, inputSeq, inputLens, hidden={'h': None}):
        embedded = self.embedding(inputSeq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputLens)
        outputs, h = self.gru(packed, hidden['h'])
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hiddenSize] + outputs[:, :, self.hiddenSize:]
        hidden = {'h': h}
        return outputs, hidden


class LstmEncoder(nn.Module):

    def __init__(self, hiddenSize, embedding, numLayers=1, dropout=0):
        super(LstmEncoder, self).__init__()
        self.hiddenSize = hiddenSize
        self.embedding = embedding
        self.numLayers = numLayers
        self.method = "lstm"
        self.lstm = nn.LSTM(hiddenSize, hiddenSize, numLayers, dropout=(0 if numLayers == 1 else dropout), bidirectional=True)
    
    def forward(self, inputSeq, inputLens, hidden=None):
        embedded = self.embedding(inputSeq)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, inputLens)
        outputs, (h, c) = self.lstm(packed) if hidden is None else self.lstm(packed, (hidden['h'], hidden['c']))
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hiddenSize] + outputs[:, :, self.hiddenSize:]
        hidden = {'h': h, 'c': c}
        return outputs, hidden