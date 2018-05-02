import argparse

from nltk.translate import bleu_score
import pickle as pkl


IS_RAW = False
class Evaluator(object):
    def __init__(self):
        self.labels = []
        self.predictions = []
        self.is_raw = IS_RAW


    def add_example(self, label_output, pred_output):
        self.labels.append(label_output)
        self.predictions.append(pred_output)

    def get_report(self):
        # find hard accuracy
        total_cnt = float(len(self.predictions))
        corr_cnt = 0

        # find entity precision, recall and f1
        tp, fp, fn = 0.0, 0.0, 0.0

        # find intent precision recall f1
        itp, ifp, ifn = 0.0, 0.0, 0.0

        # backend accuracy
        btp, bfp, bfn = 0.0, 0.0, 0.0

        # BLEU score
        refs, hyps = [], []

        for label, pred in zip(self.labels, self.predictions):
            if label == pred:
                corr_cnt += 1
            label_ent = self._get_entities(label)
            pred_ent = self._get_entities(pred)
            label_backend = self._get_backend(label_ent)
            pred_backed = self._get_backend(pred_ent)

            ttpp, ffpp, ffnn = self._get_tp_fp_fn(label_ent, pred_ent)
            tp += ttpp
            fp += ffpp
            fn += ffnn


            ttpp, ffpp, ffnn = self._get_tp_fp_fn(label_backend, pred_backed)
            btp += ttpp
            bfp += ffpp
            bfn += ffnn

            refs.append([label.split()])
            hyps.append(pred.split())

        bleu = bleu_score.corpus_bleu(refs, hyps)
        hard_accuracy = corr_cnt/(total_cnt+1e-20) # Underflow

        precision, recall, f1 = self._get_prec_recall(tp, fp, fn)
        back_precision, back_recall, back_f1 = self._get_prec_recall(btp, bfp, bfn)

        return "Hard accuracy is %f\n"                "Entity precision %f recall %f and f1 %f\n"                "Backend precision %f recall %f and f1 %f\n"                "BLEU %f\n"                % (hard_accuracy,
                  precision, recall, f1,
                  back_precision, back_recall, back_f1,
                  bleu)

    def _get_entities(self, sent):
        tokens = sent.split()
        if self.is_raw:
            entities = []
            buffer = []
            for t in tokens:
                if "<" in t:
                    entities.append(t)
                    buffer = []
                elif t.isupper() and t != "I":
                    buffer.append(t)
                elif len(buffer) > 0:
                    entities.append(" ".join(buffer))
                    buffer = []
            if len(buffer) > 0:
                entities.append(" ".join(buffer))
            # check for times
            for t in tokens:
                if t.isdigit():
                    entities.append(t)
                if "a.m" in t or "p.m" in t:
                    entities.append(t)
        else:
            entities = [t for t in tokens if "<" in t]
        return entities

    def _get_tp_fp_fn(self, label_ents, pred_ents):
        tp = len([t for t in pred_ents if t in label_ents])
        fp = len(pred_ents) - tp
        fn = len(label_ents) - tp
        return tp, fp, fn

    def _get_backend(self, entities):
        for t in entities:
            if "<backend" in t:
                return entities
        return []

    def _get_prec_recall(self, tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1


parser = argparse.ArgumentParser()
parser.add_argument('-r','--result',type=str,default='results.txt')
parser.add_argument('-t','--truth',type=str,default='truth.txt')
args = parser.parse_args()

# In[2]:


with open(args.result, 'r') as f:
    out = f.read().splitlines()


# In[3]:
with open(args.truth, 'r') as f:
    tru = f.read().splitlines()


# In[4]:
with open('test_idx.p','rb') as fin:
    test_idx = pkl.load( fin )

# In[5]:

ret = list(zip(test_idx, out))
ret.sort(key=lambda tup: tup[0])
ret = [each[1] for each in ret]


# In[6]:


ev = Evaluator()


# In[7]:


for i, each in enumerate(ret):
    ev.add_example(tru[i], each)


# In[8]:


print(ev.get_report())

