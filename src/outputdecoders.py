import globals as g
import torch
import torch.nn as nn

def buildSearchDecoder(encoder, decoder, method, topp, topk):
    if method == "greedy":
        return GreedySearchDecoder(encoder, decoder)
    elif method == "topp":
        return TopPSearchDecoder(encoder, decoder, topp)
    elif method == "topk":
        return TopKSearchDecoder(encoder, decoder, topk)
    else:
        raise ValueError("'method' must be one of: topp, topk, greedy")

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputSeq, inputLen, maxLen):
        # Forward input through encoder model
        encoderOutputs, encoderHidden = self.encoder(inputSeq, inputLen)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoderHidden = {'h': encoderHidden['h'][:self.decoder.numLayers]}
        if 'c' in encoderHidden.keys(): decoderHidden['c'] = encoderHidden['c'][:self.decoder.numLayers]

        # Initialize decoder input with SOS_token
        decoderInput = torch.ones(1, 1, device=g.device, dtype=torch.long) * g.SOS_token

        # Initialize tensors to append decoded words to
        allTokens = torch.zeros([0], device=g.device, dtype=torch.long)
        allScores = torch.zeros([0], device=g.device)

        # Iteratively decode one word token at a time
        for _ in range(maxLen):
            # Forward pass through decoder
            decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden, encoderOutputs)

            # Obtain most likely word token and its softmax score
            decoderScores, decoderInput = torch.max(decoderOutput, dim=1)

            # Record token and score
            allTokens = torch.cat((allTokens, decoderInput), dim=0)
            allScores = torch.cat((allScores, decoderScores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoderInput = torch.unsqueeze(decoderInput, 0)

            # Check to see if we've hit EOS.
            if decoderInput[0, -1] == g.EOS_token:
                break

        # Return collections of word tokens and scores
        return allTokens, allScores


class TopPSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, topP):
        super(TopPSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.topP = topP

    def forward(self, inputSeq, inputLen, maxLen):
        # Forward input through encoder model
        encoderOutputs, encoderHidden = self.encoder(inputSeq, inputLen)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoderHidden = {'h': encoderHidden['h'][:self.decoder.numLayers]}
        if 'c' in encoderHidden.keys(): decoderHidden['c'] = encoderHidden['c'][:self.decoder.numLayers]

        # Initialize decoder input with SOS_token
        decoderInput = torch.ones(1, 1, device=g.device, dtype=torch.long) * g.SOS_token

        # Initialize tensors to append decoded words to
        allTokens = torch.zeros([0], device=g.device, dtype=torch.long)
        allScores = torch.zeros([0], device=g.device)

        # Iteratively decode one word token at a time
        for _ in range(maxLen):
            # Forward pass through decoder
            decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden, encoderOutputs)

            # Pull the top vocab terms by probability up to TopP, then randomly sample a term.
            vocab, vocabToDecoderOutIndices = torch.sort(decoderOutput, dim=1, descending=True)
            sumP = 0
            iVocab = 0
            while sumP < self.topP and iVocab < len(vocab[0]):
                sumP += vocab[0][iVocab]
                iVocab += 1
            vocab = vocab[:, :iVocab]
            vocabIndex = torch.multinomial(vocab, 1) # No need to renormalize the probabilities with torch.multinomial
            decoderInput = torch.tensor([vocabToDecoderOutIndices[0, vocabIndex[0, 0]]], device=decoderOutput.device)
            decoderScores = torch.tensor([decoderOutput[0, decoderInput]], device=decoderOutput.device)

            # Record token and score
            allTokens = torch.cat((allTokens, decoderInput), dim=0)
            allScores = torch.cat((allScores, decoderScores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoderInput = torch.unsqueeze(decoderInput, 0)

            # Check to see if we've hit EOS.
            if decoderInput[0, -1] == g.EOS_token:
                break

        # Return collections of word tokens and scores
        return allTokens, allScores


class TopKSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, topK):
        super(TopKSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.topK = topK

    def forward(self, inputSeq, inputLen, maxLen):
        # Forward input through encoder model
        encoderOutputs, encoderHidden = self.encoder(inputSeq, inputLen)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoderHidden = {'h': encoderHidden['h'][:self.decoder.numLayers]}
        if 'c' in encoderHidden.keys(): decoderHidden['c'] = encoderHidden['c'][:self.decoder.numLayers]

        # Initialize decoder input with SOS_token
        decoderInput = torch.ones(1, 1, device=g.device, dtype=torch.long) * g.SOS_token

        # Initialize tensors to append decoded words to
        allTokens = torch.zeros([0], device=g.device, dtype=torch.long)
        allScores = torch.zeros([0], device=g.device)

        # Iteratively decode one word token at a time
        for _ in range(maxLen):
            # Forward pass through decoder
            decoderOutput, decoderHidden = self.decoder(decoderInput, decoderHidden, encoderOutputs)

            # Pull the top vocab terms by probability up to TopP, then randomly sample a term.
            vocab, vocabToDecoderOutIndices = torch.sort(decoderOutput, dim=1, descending=True)
            vocab = vocab[:, :self.topK]
            vocabIndex = torch.multinomial(vocab, 1) # No need to renormalize the probabilities with torch.multinomial, saves us a torch.div
            decoderInput = torch.tensor([vocabToDecoderOutIndices[0, vocabIndex[0, 0]]], device=decoderOutput.device)
            decoderScores = torch.tensor([decoderOutput[0, decoderInput]], device=decoderOutput.device)

            # Record token and score
            allTokens = torch.cat((allTokens, decoderInput), dim=0)
            allScores = torch.cat((allScores, decoderScores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoderInput = torch.unsqueeze(decoderInput, 0)

            # Check to see if we've hit EOS.
            if decoderInput[0, -1] == g.EOS_token:
                break

        # Return collections of word tokens and scores
        return allTokens, allScores