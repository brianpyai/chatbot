# -*- coding: utf-8 -*-
import re
Gbase="./"
import os,shutil,time,json
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random,time 
from transformers import PreTrainedTokenizerFast


VOCAB_PATH = Gbase + 'VOCABSE.pkl'
TOKENIZER_PATH = Gbase + 'custom_tokenize.model'
MODEL_PATH = Gbase + 'fast_embef.pt'


VOCAB_PATH = Gbase + 'VOCABSEBPE.pkl'

TOKENIZER_PATH = Gbase + 'bpe_tokenize.model'
MODEL_PATH = Gbase + 'fast_embef_bpe.pt'


MAX_LEN=1024*16
VOCAB_SIZE =256*256*2



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_or_train_tokenizer(tokenizer_path=TOKENIZER_PATH, vocab_size=VOCAB_SIZE):
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            print("Loaded existing tokenizer.")
        except ValueError:
            print("Error loading existing tokenizer. Training a new one.")
            tokenizer = train_tokenizer(wikiDict, tokenizer_path, vocab_size, keys)
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    special_tokens_dict = {
        'pad_token': '<pad>',
        'unk_token': '<unk>',
        'bos_token': '<sos>',
        'eos_token': '<eos>',
        'mask_token': '<mask>'
    }
    
    # Add special tokens one by one
    for token_type, token in special_tokens_dict.items():
        setattr(tokenizer, token_type, token)

    return tokenizer
 
class FastEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, device, tokenizer):
        super(FastEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.device = device
        self.tokenizer = tokenizer

    def forward(self, src):
        embedded = self.embedding(src)
        embedded = embedded.mean(dim=1)
        hidden = torch.relu(self.linear(embedded))
        return self.output(hidden)

    def embeds(self, text):
        with torch.no_grad():
            ids = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
            return self.embedding(ids).mean(dim=1)

    def encode(self, text):
        return self.tokenizer.encode(text)
        with torch.no_grad():
            ids = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
            #embedded = self.embedding(ids).mean(dim=1)
            #hidden = torch.relu(self.linear(embedded))
            return ids

    def decode(self, tensor):
        return self.tokenizer.decode(tensor)
        with torch.no_grad():
            output = self.output(tensor)
            return torch.argmax(output, dim=-1)


def loadFastEmbeddingModel(device=device):
    tokenizer = load_or_train_tokenizer()
    EMBED_SIZE = 128
    HIDDEN_SIZE = 1024
    vocab_size = tokenizer.vocab_size

    model = FastEmbeddingModel(vocab_size, EMBED_SIZE, HIDDEN_SIZE, device=device, tokenizer=tokenizer).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device ,weights_only=True))
        print("Loaded existing model.")
    else:
        print("No existing model found. Starting from scratch.")
    return model

EmbeddingModel = loadFastEmbeddingModel()
Embedding=lambda x :EmbeddingModel.embeds(x).tolist()[0]




alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

# 新增中文相關的正則表達式
chinese_chars = r'[\u4e00-\u9fff]'
chinese_punc = r'[。！？]'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences, supporting both English and Chinese.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if '"' in text: text = text.replace('."','".')
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    
    # 處理中文標點符號
    text = re.sub(f"({chinese_chars}+)({chinese_punc})", "\\1\\2<stop>", text)
    
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

class Text:
    Black = lambda s: "\033[30m%s\033[0m" % s
    Red = lambda s: "\033[31m%s\033[0m" % s
    Green = lambda s: "\033[32m%s\033[0m" % s
    Yellow = lambda s: "\033[33m%s\033[0m" % s
    Blue = lambda s: "\033[34m%s\033[0m" % s
    Purple = lambda s: "\033[35m%s\033[0m" % s
    Cyan = lambda s: "\033[36m%s\033[0m" % s
    Gray = lambda s: "\033[37m%s\033[0m" % s
    DarkGray = lambda s: "\033[90m%s\033[0m" % s
    LightRed = lambda s: "\033[91m%s\033[0m" % s
    LightGreen = lambda s: "\033[92m%s\033[0m" % s
    LightYellow = lambda s: "\033[93m%s\033[0m" % s
    LightBlue = lambda s: "\033[94m%s\033[0m" % s
    LightPurple = lambda s: "\033[95m%s\033[0m" % s
    LightCyan = lambda s: "\033[96m%s\033[0m" % s
    White = lambda s: "\033[97m%s\033[0m" % s


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_keywords(text, top_n=20, embedding_model=EmbeddingModel):
    lines = text.split('\n', 1)
    title = lines[0]
    content = lines[1] if len(lines) > 1 else ''
    title_tokens = embedding_model.encode(title)
    #content_tokens = embedding_model.encode(content)
    

    #if len(text )>2000:top_n=min(40,int(len(text)/200))
    #elif len(text )>20000:top_n=min(90,int(len(text)/200))
    

    title_words = embedding_model.decode(title_tokens).split()
    content_words = split_into_sentences(content)
    top_n=min ( 100 ,max(20,int( len(content_words)/33)+len(title_words) ) )
    #embedding_model.decode(content_tokens).split()

    word_freq = {}
    for word in title_words:
        word_freq[word] = word_freq.get(word, 0) + 3  
    for word in content_words:
        word_freq[word] = word_freq.get(word, 0) + 1

 
    total_words = len(title_words) * 3 + len(content_words)
    word_scores = []
    for word, freq in word_freq.items():
        tf = freq / total_words
        idf = 1  
        score = tf * idf
        word_scores.append((word, score))


    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in word_scores[:top_n]]


def build_relationship_tree(keywords, embeddings, threshold=0.55):
    tree = {}
    for i, keyword1 in enumerate(keywords):
        tree[keyword1] = []
        for j, keyword2 in enumerate(keywords[i+1:]):
            similarity = cosine_similarity([embeddings(keyword1)], [embeddings(keyword2)])[0][0]
            if similarity > threshold:
                tree[keyword1].append((keyword2, similarity))
        tree[keyword1].sort(key=lambda x: x[1], reverse=True)
    return tree

def print_tree(tree, root, level=0, max_children=4):
    prefix = "  " * level
    result = ""
    #result += f"{prefix}{root}\n"
    if root in tree:
        for i, (child, similarity) in enumerate(tree[root][:max_children]):
            result += f"{prefix}├─ {child} ({similarity:.2f})\n"
            if i < len(tree[root]) - 1:
                result += print_tree(tree, child, level + 1, max_children)
            else:
                result += print_tree(tree, child, level + 1, max_children)
    return result

def text_relationship_tree(text, embedding_model=Embedding):
    keywords = extract_keywords(text)
    tree = build_relationship_tree(keywords, embedding_model)
    
    #print("Text Relationship Tree:")
    tree_str = print_tree(tree, keywords[0])
    print(tree_str)
    return tree_str



from fileDict3 import FileDict 
wiki=FileDict("wikipedia.sql3")

if __name__ == '__main__':
    import random 
    random.seed(time.time())    
    items =wiki.items()
    n=800*100*100
    f=open("testBuildTree.txt","w" ,encoding="utf-8")
    #its=[]
    m=0
    for x in range (n):
        k,v=next(items)
        if random.randint(1,30000)!=10 :continue 
        m+=1
        if m>100 :break
        
        its=[]
        text=k +"\n"+v
        #print (k)
        s=text_relationship_tree(text)
        f.write(k+":\n"+s+"\n")
    f.close()
        