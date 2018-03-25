import pickle as pkl
from collections import defaultdict
import numpy as np

class LetsGoCorpus(object):

    def __init__(self, data_path):
        train, valid, test = pkl.load(open(data_path, "rb"), encoding='bytes')
        self.train = self.load_data(train)
        self.valid = self.load_data(valid)
        self.test = self.load_data(test)
        print("Loaded %d train %d valid and %d test" % (len(self.train), len(self.valid), len(self.test)))

    def load_data(self, data):
        ret_dial = []
        for dial in data:
            ret_turn = []
            for turn in dial:
                sys = turn[0]
                usr = turn[1]
                try:
                    sys = sys.decode()
                except:
                    pass
                try:
                    usr = usr.decode()
                except:
                    pass
                ret_turn.append((sys, usr, turn[2], turn[3]))

            ret_dial.append(ret_turn)
        return ret_dial

    def get_train_sents(self):
        sys_sents = []
        usr_sents = []
        for dial in self.train:
            for turn in dial:
                sys_sents.append(turn[0])
                usr_sents.append(turn[1])

        return sys_sents + usr_sents, sys_sents

def read_corpus_vocab(corp, source):
    data = []
    for line in corp:
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data




def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle: np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents

# Data feed
class DataLoader(object):
    batch_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    indexes = None
    data_size = None
    name = None
    equal_len_batch = None
    # if equal batch, the indexes sorted by data length
    # else indexes = range(data_size)

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, selected_indexes):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle=True):
        if self.name.upper() == "TEST":
            new_batch_size = None
            for i in range(3):
                factors = utils.factors(self.data_size-i)
                temp = min(factors, key=lambda x:abs(x-batch_size))
                if np.abs(temp-batch_size) < batch_size * 0.5:
                    new_batch_size = temp
                    break
            if new_batch_size is not None:
                batch_size = new_batch_size
                print("Adjust the batch size to %d" % batch_size)

        self.ptr = 0
        self.batch_size = batch_size
        self.num_batch = self.data_size // batch_size
        print("Number of left over sample %d" % (self.data_size-batch_size*self.num_batch))

        # if shuffle and we don't want to group lines, shuffle index
        if shuffle and not self.equal_len_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        # if shuffle and we want to group lines, shuffle batch indexes
        if shuffle and self.equal_len_batch:
            self._shuffle_batch_indexes()

        print("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_indexes=selected_ids)
        else:
            return None


class SimpleLetsGoDataLoader(DataLoader):
    def __init__(self, name, data_x, data_y, equal_batch, config):
        self.equal_len_batch = equal_batch
        self.name = name
        self.data_x = data_x
        self.data_y = data_y
        self.data_size = len(self.data_y)
        self.max_sys_len = config.max_sys_len
        self.max_usr_len = config.max_usr_len

        all_lens = [len(line) for line in self.data_y]
        all_usr_lens = [len(line[1]) for ctx in self.data_x for line in ctx]

        print("Sys: Max len %d and min len %d and avg len %f" %
              (np.max(all_lens), np.min(all_lens), float(np.mean(all_lens))))
        print("Usr: Max len %d and min len %d and avg len %f" %
              (np.max(all_usr_lens), np.min(all_usr_lens), float(np.mean(all_usr_lens))))
        if equal_batch:
            self.indexes = list(np.argsort(all_lens))
        else:
            self.indexes = range(len(self.data_y))

    def _prepare_batch(self, selected_indexes):
        selected_xs = [self.data_x[i] for i in selected_indexes]
        selected_ys = [self.data_y[i] for i in selected_indexes]

        # get response vectors
        y_lens = [len(y) for y in selected_ys]
        max_y_len = np.max(y_lens)

        # context
        c_lens = [len(x) for x in selected_xs]
        max_c_len = np.max(c_lens)

        # vectors
        context_sys_vec = np.zeros((self.batch_size, max_c_len, self.max_sys_len), dtype=np.int32)
        context_usr_vec = np.zeros((self.batch_size, max_c_len, self.max_usr_len), dtype=np.int32)
        context_score_vec = np.zeros((self.batch_size, max_c_len, 1), dtype=np.float32)
        y_vec = np.zeros((self.batch_size, max_y_len), dtype=np.int32)

        for b_id in range(self.batch_size):
            # for context
            for c_id in range(max_c_len):
                if c_id < c_lens[b_id]:
                    turn = selected_xs[b_id][c_id]
                    truncat_sys = turn[0][0:self.max_sys_len]
                    truncat_usr = turn[1][0:self.max_usr_len]

                    context_sys_vec[b_id, c_id, 0:len(truncat_sys)] = truncat_sys
                    context_usr_vec[b_id, c_id, 0:len(truncat_usr)] = truncat_usr
                    context_score_vec[b_id, c_id, 0] = turn[2]

            y_vec[b_id, 0:y_lens[b_id]] = selected_ys[b_id]

        return context_sys_vec, context_usr_vec, context_score_vec, c_lens, y_vec, y_lens

class LetsGoDataLoader():
    def __init__(self, data):

        self.data = data
        self.src = []
        self.tgt = []

        self.process()

    def process(self):
        for dial in self.data:
            for i, turn in enumerate(dial):
                src_ctx = []
                if i == 0: continue

                for prev in dial[:i]:
                    sys = prev[0].strip().split(' ')
                    usr = prev[1].strip().split(' ')
                    src_ctx.append((sys, usr, prev[2], prev[3]))

                self.src.append(src_ctx)

                self.tgt.append(['<s>'] + turn[0].strip().split(' ') + ['</s>'])

    def get_src(self):
        return self.src

    def get_tgt(self):
        return self.tgt


class FakeLetsGoDataLoader():
    def __init__(self, data):

        self.data = data
        self.src = []
        self.tgt = []

        self.process()

    def process(self):
        for dial in self.data:
            for i, turn in enumerate(dial):
                src_ctx = []
                if i == 0: continue

                for prev in dial[:i]:
                    sys = prev[0].strip().split(' ')
                    usr = prev[1].strip().split(' ')
                    src_ctx.append(sys + usr)

                self.src.append(src_ctx)

                self.tgt.append(['<s>'] + turn[0].strip().split(' ') + ['</s>'])

    def get_src(self):
        return self.src

    def get_tgt(self):
        return self.tgt



























