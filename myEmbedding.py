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

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer



def extract_keywords(text, top_n=20, embedding_model=EmbeddingModel,split_sentences=False):
    lines = text.split('\n', 1)
    title = lines[0]
    content = lines[1] if len(lines) > 1 else ''
    title_tokens = embedding_model.encode(title)
    content_tokens = embedding_model.encode(content)
    

    #if len(text )>2000:top_n=min(40,int(len(text)/200))
    #elif len(text )>20000:top_n=min(90,int(len(text)/200))
    

    title_words = embedding_model.decode(title_tokens).split()
    content_words=split_into_sentences(content) if split_sentences else embedding_model.decode(content_tokens).split()  #
    top_n=min ( 100 ,max(20,int( len(content_words)/33)+len(title_words) ) )
    

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
    l=int(len (word_scores)*4/10)
    return [word for word, score in word_scores][:l]


def build_relationship_tree(keywords, embeddings):
    tree = {keyword: [] for keyword in keywords}
    all_similarities = []
    
    # 計算所有關鍵詞對之間的相似度
    for i, keyword1 in enumerate(keywords):
        for j, keyword2 in enumerate(keywords[i+1:], start=i+1):
            similarity = cosine_similarity([embeddings(keyword1)], [embeddings(keyword2)])[0][0]
            all_similarities.append((keyword1, keyword2, similarity))
    
    # 計算相似度閾值
    all_similarities.sort(key=lambda x: x[2], reverse=True)
    num_relations = len(all_similarities)
    num_to_keep = min(num_relations, min(80, max(int(num_relations / 50), 10)))
    
    if num_to_keep > 0:
        similarity_threshold = all_similarities[num_to_keep-1][2]
    else:
        similarity_threshold = 0  # 如果沒有關係，設置閾值為0
    
    # 根據閾值添加關係，保持原始順序
    for i, keyword1 in enumerate(keywords):
        for j, keyword2 in enumerate(keywords[i+1:], start=i+1):
            similarity = cosine_similarity([embeddings(keyword1)], [embeddings(keyword2)])[0][0]
            if similarity >= similarity_threshold:
                tree[keyword1].append((keyword2, similarity))
                tree[keyword2].append((keyword1, similarity))
    
    # 對每個關鍵詞的關係列表進行排序
    for keyword in tree:
        tree[keyword].sort(key=lambda x: x[1], reverse=True)
    
    return tree
def print_tree(tree, root_keyword, max_depth=4):
    result = ""
    stack = [(root_keyword, 0, "")]
    printed = set()
    
    while stack:
        keyword, depth, prefix = stack.pop()
        if depth > max_depth or keyword in printed:
            continue
        
        #result += f"{prefix}{keyword}\n"
        #printed.add(keyword)
        
        if keyword in tree:
            children = tree[keyword]
            for i, (child, similarity) in enumerate(reversed(children)):
                if child not in printed:
                    if i == 0:
                        new_prefix = prefix + "  "
                        child_prefix = prefix + "└ "
                    else:
                        new_prefix = prefix + "│ "
                        child_prefix = prefix + "├ "
                    result += f"{child_prefix}{child} ({similarity:.2f})\n"
                    stack.append((child, depth + 1, new_prefix))
    
    return result

def text_relationship_tree(text, embedding_model=Embedding,split_sentences=False):
    keywords=extract_keywords(text,split_sentences=split_sentences)
    
    
    tree = build_relationship_tree(keywords, embedding_model)
    
    #print("Text Relationship Tree:")
    try :
        tree_str = print_tree(tree, keywords[0])
        print(tree_str)
        return tree_str
    except :
        import traceback
        traceback.print_exc()
        return "" 



from fileDict3 import FileDict 
wiki=FileDict("wikipedia.sql3")

if __name__ == '__main__':
    import random 
    random.seed(time.time())    
    items =wiki.items()
    n=800*100*100
    
    f1=open("testBuildTree.txt" , "w" ,encoding="utf-8")
    f2=open("testBuildTreeSentence.txt" , "w" ,encoding="utf-8")
    m=0
    for x in range (n):
        k,v=next(items)
        if random.randint(1,30000)!=10 :continue 
        m+=1
        if m>20 :break
        
        its=[]
        text=k +"\n"+v
        #print (k)
        s1=text_relationship_tree(text,split_sentences=False)
        s2=text_relationship_tree(text,split_sentences=True)
        f1.write(k+":\n"+s1+"\n")
        f2.write(k+":\n"+s2+"\n")
    f1.close()
    f2.close()
    
        
