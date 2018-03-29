from __future__ import division
import torch
import torch.nn.functional as F

"""
 Class for managing the internals of the beam search process.
         hyp1-hyp1---hyp1 -hyp1
                 \             /
         hyp2 \-hyp2 /-hyp2hyp2
                               /      \
         hyp3-hyp3---hyp3 -hyp3
         ========================
 Takes care of beams, back pointers, and scores.
"""


class BeamOld(object):
    def __init__(self, size, cuda=False):

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        # Fill all with PAD
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        # First is <start>
        self.nextYs[0][0] = 1

        # The attentions (matrix) for each time.
        self.attn = []

        # yulun
        self.counter = 0
        self.result = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            beamLk = wordLk[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        self.attn.append(attnOut.index_select(0, prevK))

        # # End condition is when top-of-beam is EOS.
        # if self.nextYs[-1][0] == 2:
        #     self.done = True
        #     self.allScores.append(self.scores)
        for i, id in enumerate(self.nextYs[-1].cpu()):
            if id == 2:
                self.result.append((len(self.nextYs) - 1, i))


        if len(self.result) == self.size:
            self.done = True
            self.allScores.append(self.scores)
            # print self.result

        # print self.nextYs
        # print self.scores
        # print self.prevKs
        # print self.allScores
        # print 'haha'

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    def getBest(self):
        "Get the score of the best in the beam."
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def getHyp(self, k):
        """
        Walk back to construct the full hypothesis.
        Parameters.
             * `k` - the position in the beam to construct.
         Returns.
            1. The hypothesis
            2. The attention at each time step.
        """
        hyp, attn = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]

        return hyp[::-1], torch.stack(attn[::-1])

    def getAllHyp(self):

        hyps, attns = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for i, k in self.result:
            hyp, attn = [], []
            for j in range(i - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                attn.append(self.attn[j][k])
                k = self.prevKs[j][k]

            hyps.append(hyp[::-1])
            attns.append(torch.stack(attn[::-1]))

        return [hyps], attns

class Beam(object):
    def __init__(self, size, n_best=5, cuda=False):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = 1

        # Has EOS topped the beam yet.
        self._eos = 2
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best


    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == 2:
            # self.allScores.append(self.scores)
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.n_best

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]
        return hyp[::-1], torch.stack(attn[::-1])

class RLBeam(object):
    def __init__(self, size, n_best=5, cuda=False):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = 1

        # Has EOS topped the beam yet.
        self._eos = 2
        self.eosTop = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        '''for RL stuff.'''
        # sample loss at each timestamp
        self.nextLs = [self.tt.LongTensor(size).fill_(0)]
        # baseline at each timestamp
        self.nextBs = [self.tt.LongTensor(size).fill_(0)]
        # mask at each timestamp
        self.nextMs = [self.tt.LongTensor(size).fill_(0)]


    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk, attnOut, baseline, hidden_t):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        self.attn.append(attnOut.index_select(0, prevK))

        '''compute rl sample and baseline losses here'''
        sample_p_t = torch.gather(wordLk, 1, self.nextYs[-1].unsqueeze(1)).squeeze(1)
        self.nextLs.append(-sample_p_t)

        b_t = F.tanh(baseline(hidden_t.detach()))
        self.nextBs.append(b_t)

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == 2:
            # self.allScores.append(self.scores)
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.n_best

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        sample_losses, baselines = [], []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])

            ''' rl stuff '''
            sample_losses.append(self.nextLs[j+1][k])
            baselines.append(self.nextBs[j+1][k])

            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]

        return hyp[::-1], torch.stack(attn[::-1]), torch.FloatTensor(sample_losses[::-1]), torch.stack(baselines[::-1])