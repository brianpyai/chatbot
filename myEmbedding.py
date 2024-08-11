
Gbase="./"
import os,shutil,time,json
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
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

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    word_scores = list(zip(feature_names, tfidf_scores))
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in word_scores[:top_n]]

def build_relationship_tree(keywords, embeddings, threshold=0.5):
    tree = {}
    for i, keyword1 in enumerate(keywords):
        tree[keyword1] = []
        for j, keyword2 in enumerate(keywords[i+1:]):
            similarity = cosine_similarity([embeddings(keyword1)], [embeddings(keyword2)])[0][0]
            if similarity > threshold:
                tree[keyword1].append((keyword2, similarity))
        tree[keyword1].sort(key=lambda x: x[1], reverse=True)
    return tree

def print_tree(tree, root, level=0, max_children=3):
    prefix = "  " * level
    print(f"{prefix}{root}")
    if root in tree:
        for i, (child, similarity) in enumerate(tree[root][:max_children]):
            print(f"{prefix}├─ {child} ({similarity:.2f})")
            if i < len(tree[root]) - 1:
                print_tree(tree, child, level + 1, max_children)
            else:
                print_tree(tree, child, level + 1, max_children)

def text_relationship_tree(text, embedding_model=Embedding):
    keywords = extract_keywords(text)
    tree = build_relationship_tree(keywords, embedding_model)
    
    print("Text Relationship Tree:")
    print_tree(tree, keywords[0])


from fileDict3 import FileDict 
wiki=FileDict("wikipedia.sql3")

if __name__ == '__main__':
    import random 
    text="script 很早就是翻譯「腳本」，戲劇圈也是這樣講。腳本劇本都行。抓支語抓錯在所難免，辨證後，接受它是台灣既有用語即可。台灣的「排支運動」僅限於台灣，只對台灣正體中文重要，跟其他簡體字中文語言區無關。原因無它，簡體中文和台灣正體中文早已是兩種語言。台灣不能接受鳳梨被講成波蘿、也不會接受馬玲薯被改成土豆；不能接受 橫列直行 曲解成 直列橫行.在台灣有台灣標準答案的，錯就錯、對就對；我看不懂留言區一堆支語支持者在嗨什麼。.#2024年国漫开分TOP3# 《眷思量2》开分9.0、《仙逆》年番开分9.4、《诛仙2》开分9.3！2024上半年高分国漫TOP3有了！《仙逆》认真看用户的评论听取意见修改，王麻子的建模一次比一次惊艳，尤其是云天宗抢亲，这波给我甜麻了！《诛仙》第二季质感在线，打戏特效很燃，陆雪琪和鬼厉雨夜重逢还有花海大战狠狠把我刀没了！《眷思量》第二季镜玄出手偷偷救屠丽，主角团团战打戏配合默契，每个角色的高光时刻都特别酷！一句话总结，你们仨值得！"
    print(Text.Yellow (EmbeddingModel.encode(text)) )
    
    print( Text.Cyan (EmbeddingModel.decode(EmbeddingModel.encode(text)) ) )
    print (text)
    t0=time.time()
    n=1000
    for x in range (n):
        embed =Embedding(text)
    print (embed[:10]  )
    print (Text.Cyan ( " %s %s %s " % ( len(embed) , n*len (text )/(time.time ()-t0) ,"/s") ) )
    items =wiki.items()
    n=800*100*100
    for x in range (n):
        k,v=next(items)
        if random.randint(0,n)<100:continue 
        text=k +"\n"+v
        print (k)
        text_relationship_tree(text)
        