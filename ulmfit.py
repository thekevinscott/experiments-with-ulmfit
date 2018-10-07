import sys
sys.path.append('../')
from fastai.text import *
import html
import os
from sklearn.model_selection import train_test_split
import concurrent.futures
import spacy
import time
import torch.nn as nn


def gather_texts(path):
    """
    This function will go through the aclImdb folder
    and create the necessary datasets
    """
    
    # initializes the text and labels collections
    texts,labels = [],[]

    
    # for each sentiment
    for idx,label in enumerate(CLASSES):
    
    
        # will go through the 
        for fname in (path/label).glob('*.*'):
    
    
            # open the file and append the filetext
            texts.append(fname.open('r').read())
    
    
            # open 
            labels.append(idx)
    return np.array(texts),np.array(labels)

def proc_all_mp(ss, lang='en'):
        ncpus = num_cpus()//2
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang]*len(ss)), [])

def partition_by_cores(a):
    return partition(a, len(a)//num_cpus() + 1)

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, prime))

def fixup(x):
    """ Cleans up erroroneus characters"""
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def get_texts(df, n_lbls=1):
    # pull the labels out from the dataframe
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    
    # pull the full FILEPATH for each text
    # BOS is a flag to indicate when a new text is starting
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    
    # Sometimes, text has title, or other sub-sections. We will record all of these
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = texts.apply(fixup).values.astype(str)

    # Tokenize the data
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


class LanguageModelLoader():
    """Returns tuples of mini-batches."""
    
    def __init__(self, nums, bs, bptt, backwards=False):
        
        # assign values
        self.bs,self.bptt,self.backwards = bs,bptt,backwards
        
        # batchify the numbers. Based on the batchsize
        # subdivide the data. Note: batchsize 64
        # 640,000 would be broken into 64 x 10,000 
        self.data = self.batchify(nums)
        
        # initialize other values
        self.i,self.iter = 0,0
        self.n = len(self.data)


    def __iter__(self):
        """ Iterator implementation"""
            
        # start from zero
        self.i,self.iter = 0,0
            
        # will continually pull data out
        while self.i < self.n-1 and self.iter<len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
                        
            # yields the value
            yield res

                        
    def __len__(self): return self.n // self.bptt - 1

                        
    def batchify(self, data):
        """splits the data into batch_size counts of sets"""
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards: data=data[::-1]
                                
        # returns the transpose
        # have batch_size number of columns 
        return T(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)
    
class LanguageModelData():
    """
    - a training data loader
    - a validation data loader
    - a test loader
    - a saving path
    - model parameteres
    """
    def __init__(self, path, pad_idx, nt, trn_dl, val_dl, test_dl=None, bptt=70, backwards=False, **kwargs):
        self.path,self.pad_idx,self.nt = path,pad_idx,nt
        self.trn_dl,self.val_dl,self.test_dl = trn_dl,val_dl,test_dl

    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        m = get_language_model(self.nt, emb_sz, n_hid, n_layers, self.pad_idx, **kwargs)
        model = LanguageModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)
    
class LanguageModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [*zip(m.rnns, m.dropouths), (self.model[1], m.dropouti)]
    
class RNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.cross_entropy

    def save_encoder(self, name): save_model(self.model[0], self.get_model_path(name))
    def load_encoder(self, name): load_model(self.model[0], self.get_model_path(name))
    

class RNN_Encoder(nn.Module):

    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM layers to drive the network, and
        - variational dropouts in the embedding and LSTM layers
        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange=0.1

    def __init__(self, ntoken, emb_sz, nhid, nlayers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5):
        """ Default constructor for the RNN_Encoder class
            Args:
                bs (int): batch size of input data
                ntoken (int): number of vocabulary (or tokens) in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                nhid (int): number of hidden activation per LSTM layer
                nlayers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
            Returns:
                None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs = 1
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else nhid, (nhid if l != nlayers - 1 else emb_sz)//self.ndir,
             1, bidirectional=bidir, dropout=dropouth) for l in range(nlayers)]
        if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.emb_sz,self.nhid,self.nlayers,self.dropoute = emb_sz,nhid,nlayers,dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(nlayers)])

    def forward(self, input):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)
        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()

        emb = self.encoder_with_dropout(input, dropout=self.dropoute if self.training else 0)
        emb = self.dropouti(emb)

        raw_output = emb
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,drop) in enumerate(zip(self.rnns, self.dropouths)):
            current_input = raw_output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1: raw_output = drop(raw_output)
            outputs.append(raw_output)

        self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.nhid if l != self.nlayers - 1 else self.emb_sz)//self.ndir
        return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.nlayers)]



"""
when given to pandas, it won't return a full dataframe, 
but it will return an iterator. It will return sub-sized chunks
over and over again
"""
chunksize=24000
            
re1 = re.compile(r'  +')
nlp = spacy.load('en')

# create paths to save future features
CLAS_PATH=Path('data/imdb_clas/')
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH=Path('data/imdb_lm/')
LM_PATH.mkdir(exist_ok=True)

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
CLASSES = ['neg', 'pos', 'unsup']

PATH=Path('data/aclImdb/')

if os.path.isfile(LM_PATH/'train.csv'):
    df_trn = pd.read_csv(LM_PATH/'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(LM_PATH/'test.csv', header=None, chunksize=chunksize)
else:
    trn_texts,trn_labels = gather_texts(PATH/'train')
    val_texts,val_labels = gather_texts(PATH/'test')

    col_names = ['labels','text']
    np.random.seed(42)


    # shuffle the indexes in place
    trn_idx = np.random.permutation(len(trn_texts))
    val_idx = np.random.permutation(len(val_texts))


    #shuffle texts
    trn_texts = trn_texts[trn_idx]
    val_texts = val_texts[val_idx]


    #shuffle the labels 
    trn_labels = trn_labels[trn_idx]
    val_labels = val_labels[val_idx]


    #create dataframe
    df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
    df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)


    # saving training and validation dataset
    df_trn[df_trn['labels']!=2].to_csv(CLAS_PATH/'train.csv', header=False, index=False)
    df_val.to_csv(CLAS_PATH/'test.csv', header=False, index=False)


    #write the classes
    (CLAS_PATH/'classes.txt').open('w').writelines(f'{o}\n' for o in CLASSES)

    trn_texts,val_texts = train_test_split(np.concatenate([trn_texts,val_texts]), test_size=0.1)

    df_trn = pd.DataFrame({'text':trn_texts, 'labels':[0]*len(trn_texts)}, columns=col_names)
    df_val = pd.DataFrame({'text':val_texts, 'labels':[0]*len(val_texts)}, columns=col_names)

    df_trn.to_csv(LM_PATH/'train.csv', header=False, index=False)
    df_val.to_csv(LM_PATH/'test.csv', header=False, index=False)
    
    df_trn = pd.read_csv(LM_PATH/'train.csv', header=None, chunksize=chunksize)
    df_val = pd.read_csv(LM_PATH/'test.csv', header=None, chunksize=chunksize)


__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

if os.path.isfile(LM_PATH/'tmp'/'tok_trn.npy'):
    tok_trn = np.load(LM_PATH/'tmp'/'tok_trn.npy')
    tok_val = np.load(LM_PATH/'tmp'/'tok_val.npy')
else:

    tok_trn, trn_labels = get_all(df_trn, 1)
    tok_val, val_labels = get_all(df_val, 1)

    (LM_PATH/'tmp').mkdir(exist_ok=True)

    np.save(LM_PATH/'tmp'/'tok_trn.npy', tok_trn)
    np.save(LM_PATH/'tmp'/'tok_val.npy', tok_val)

freq = Counter(p for o in tok_trn for p in o)

max_vocab = 60000
min_freq = 2

if os.path.isfile(LM_PATH/'tmp'/'itos.pkl'):
    trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
    val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
    itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))
else:
    # index to word
    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')


    # word to index
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    print(len(itos))


    # create an array of token_indices 
    trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_lm = np.array([[stoi[o] for o in p] for p in tok_val])


    # save the i
    np.save(LM_PATH/'tmp'/'trn_ids.npy', trn_lm)
    np.save(LM_PATH/'tmp'/'val_ids.npy', val_lm)
    pickle.dump(itos, open(LM_PATH/'tmp'/'itos.pkl', 'wb'))
    trn_lm = np.load(LM_PATH/'tmp'/'trn_ids.npy')
    val_lm = np.load(LM_PATH/'tmp'/'val_ids.npy')
    itos = pickle.load(open(LM_PATH/'tmp'/'itos.pkl', 'rb'))


drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7
    
vs=len(itos)


em_sz,nh,nl = 400,1150,3

# set filepaths
PRE_PATH = PATH/'models'/'wt103'
PRE_LM_PATH = PRE_PATH/'fwd_wt103.h5'


# load weights (returns a dictionary)
wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)



# pull out embedding weights
# sized vocab x em_sz 
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)


# load pre trained vocab to index mappings
itos2 = pickle.load((PRE_PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})


# create a pre-trained -> current corpus vocab to vocab mapping
# initialize an empty matrix
new_w = np.zeros((vs, em_sz), dtype=np.float32)


# loop through by row index and insert the correct embedding
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m

    
# create our torch `state` that we will load later
wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))

wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7

learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.unfreeze()

learner.model.load_state_dict(wgts)

# fit a single cycle
lr=1e-3
lrs = lr
start = time.time()
learner.fit(lrs/2, 1, wds=wd, use_clr=(32,2), cycle_len=1)
print("time to train 1 epoch,", time.time()-start)

learner.save('lm_last_ft')
learner.load('lm_last_ft')
learner.unfreeze()

# search for a learning rate, then run for 15 epoches
learner.lr_find(start_lr=lrs/10, end_lr=lrs*10, linear=True)
learner.sched.plot()

start = time.time()
learner.fit(lrs, 1, wds=wd, use_clr=(20,10), cycle_len=15)
print("Time to train,", time.time() - start)

# saves the model
learner.save('lm1')

# saves just the RNN encoder (rnn_enc)
learner.save_encoder('lm1_enc')
learner.sched.plot_loss()

# read in the data again
df_trn = pd.read_csv(CLAS_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(CLAS_PATH/'test.csv', header=None, chunksize=chunksize)


