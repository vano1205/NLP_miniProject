import nltk;
from nltk.corpus import wordnet;
from bs4 import BeautifulSoup;
from nltk.tokenize import sent_tokenize,word_tokenize
from wiktionaryparser import WiktionaryParser
from pywsd import disambiguate
import requests

raw=''
req  = requests.get('http://shakespeare.mit.edu/')
html = req.text
soup = BeautifulSoup(html, 'html.parser')
subjects=[]
links=soup.findAll('a')
for link in links:
    if 'href' in link.attrs:
        if 'Poetry' not in link.attrs['href']:
            link.attrs['href']=link.attrs['href'].replace('index','full')
        subjects.append(link.attrs['href'])

# access each literature
for sub in subjects[2:-2]:
    req  = requests.get('http://shakespeare.mit.edu/' + sub)
    html = req.text
    raw += BeautifulSoup(html, 'html.parser').get_text()

sent_token = sent_tokenize(raw)
parser=WiktionaryParser()

prondict = nltk.corpus.cmudict.dict()

# non pattern heteronym list
heteronym={}
# pattern heteronym list
pos_heteronym={}
#step one: retrieve list of heteronyms
for dictword in prondict:
    pronlist =prondict[dictword]
    wordsyns=wordnet.synsets(dictword)
    if len(pronlist)==1:
        continue
    else:
        for proninstancelist in pronlist:
            for phoneme in proninstancelist:
                if phoneme[-1]=='1':
                    for otherinstance in (otherlist for otherlist in pronlist if proninstancelist!=otherlist):
                        for otherphoneme in (entry for entry in otherinstance if entry[-1]=='1'):
                            # heteronym with different stress placement and different pos ex) PROduce, proDUCE -> "pattern heteronym"
                            if (proninstancelist.index(phoneme) +1 <= 0.5* len(proninstancelist) and otherinstance.index(otherphoneme)+1 > 0.5*len(otherinstance)) or (proninstancelist.index(phoneme) +1 > 0.5* len(proninstancelist) and otherinstance.index(otherphoneme)+1 <= 0.5*len(otherinstance)):
                                if len(wordsyns)>1 and len(proninstancelist)==len(otherinstance):
                                    for i in range(len(wordsyns)):
                                        if wordsyns[0].pos()!=wordsyns[i].pos() and not ((wordsyns[0].pos()=='a' and wordsyns[i].pos()=='s') or (wordsyns[0].pos()=='s' and wordsyns[i].pos()=='a')):
                                            posdict={}
                                            # noun,verb - noun: stress at front, verb: stress at back
                                            if (wordsyns[0].pos()=='n' and wordsyns[i].pos()=='v') or (wordsyns[0].pos()=='v' and wordsyns[i].pos()=='n'):
                                                if (proninstancelist.index(phoneme) +1 < 0.5* len(proninstancelist) and otherinstance.index(otherphoneme)+1 > 0.5*len(otherinstance)):     
                                                    posdict['N']=proninstancelist
                                                    posdict['V']=otherinstance
                                                    posdict['J']=[]

                                                    # the word has adjective as well - stress at front ex) present
                                                    for adj_checker in wordsyns:
                                                        if adj_checker.pos()=='a' or adj_checker.pos()=='s':
                                                            posdict['N']=proninstancelist
                                                            posdict['V']=otherinstance
                                                            posdict['J']=proninstancelist
                                                else:
                                                    posdict['N']=otherinstance
                                                    posdict['V']=proninstancelist
                                                    posdict['J']=[]
                                                    
                                                    for adj_checker in wordsyns:
                                                        if adj_checker.pos()=='a' or adj_checker.pos()=='s':
                                                            posdict['N']=otherinstance
                                                            posdict['V']=proninstancelist
                                                            posdict['J']=otherinstance
                                            
                                            # adjective,verb - adjective: stress at front, verb: stress at back
                                            elif ((wordsyns[0].pos()=='a' or wordsyns[0].pos()=='s') and wordsyns[i].pos()=='v') or (wordsyns[0].pos()=='v' and (wordsyns[i].pos()=='a' or wordsyns[i].pos()=='s')):
                                                if (proninstancelist.index(phoneme) +1 < 0.5* len(proninstancelist) and otherinstance.index(otherphoneme)+1 > 0.5*len(otherinstance)):
                                                    posdict['J']=proninstancelist
                                                    posdict['V']=otherinstance
                                                    posdict['N']=[]
                                                else:
                                                    posdict['J']=otherinstance
                                                    posdict['V']=proninstancelist
                                                    posdict['N']=[]

                                            if len(posdict)>0:        
                                                pos_heteronym[dictword]=posdict
                                            continue
                                continue

                            # "nonpattern heteronym" ex)wind,tear,bow,bass...
                            if phoneme[0]!=otherphoneme[0]:
                                # the stress has to be in same position
                                if proninstancelist.index(phoneme)==otherinstance.index(otherphoneme) and len(proninstancelist)==len(otherinstance) and proninstancelist.index(phoneme)!=0:
                                    temppron=[x for x in proninstancelist if x!=phoneme]
                                    tempother=[y for y in otherinstance if y!=otherphoneme]
                                    # only one should be different
                                    if temppron==tempother:
                                        if len(wordsyns)>1:
                                            cmp=0
                                            for i in range(len(wordsyns)):
                                                # has to have at least 2 meanings
                                                if wordsyns[cmp].pos() == wordsyns[i].pos():
                                                    similarity=wordsyns[cmp].path_similarity(wordsyns[i])
                                                    if similarity==None or similarity<=0.07:
                                                        heteronym[dictword]=pronlist
                                                        continue
                                                else:
                                                    cmp=i


dictfile=open('HeteroDic.txt','w',newline="")
heteronym_dict={}

# furthur filter heteronym
heteronym_list=list(heteronym)
for heteroword in heteronym:
    dictentry=parser.fetch(heteroword)

    # word not in dictionary
    for wordsense in dictentry:
        ipa_list=[]
        if len(wordsense['pronunciations']['text'])==0:
            dictentry.remove(wordsense)
        else:
            # no IPA pronunciation is excluded
            for iterator in wordsense['pronunciations']['text']:
                if 'IPA' in iterator:
                    ipa_list.append(iterator.split('IPA:')[-1])
            if len(set(ipa_list))<2:
                dictentry.remove(wordsense)
   
    #single meaning excluded
    if len(dictentry)<=1:
        continue
    heteronym_dict[heteroword]=dictentry
    dictfile.write(heteroword+str(dictentry)+'\n')


def pos_convert_tag (posinfo):
    if posinfo=='verb':
        return 'VB'
    elif posinfo=='noun':
        return 'NN'
    elif posinfo=='adjective':
        return 'JJ'
    elif posinfo=='adverb':
        return 'RB'
    else:
        return ''

# step3: map meaning to pronunciation
def map_meaning2pronun (synlist,targetword,targettag):
    dictinfo=heteronym_dict[targetword]
    max_common_word=0
    # the index that has max common words (dictionary)
    max_defindex=-1

    for indexiter, dictinstance in enumerate(dictinfo):
        definitions=dictinstance['definitions']
        for def_unique_pos in definitions:
            posinfo=def_unique_pos['partOfSpeech']
            cmpmeaning=def_unique_pos['text']
            # differ pos 
            if pos_convert_tag(posinfo) not in targettag:
                continue
            # find the definition that has most intersection
            if len(synlist)!=0:
                for synentry in synlist:
                    stdmeaning=synentry.definition()
                    stdset = set(word_tokenize(str(stdmeaning)))
                    cmplist=[]
                    for small_meaning in cmpmeaning:
                        cmplist+=word_tokenize(small_meaning)
                    cmpset=set(cmplist)
                    intersectset=cmpset.intersection(stdset)
                    intersectnum=len(intersectset)
                    if intersectnum>max_common_word:
                        max_common_word=intersectnum
                        max_defindex=indexiter

    pronunlist_dict= dictinfo[0]['pronunciations']['text']

    ipa_list=[]
    enPR_list=[]
    for iterator in pronunlist_dict:
        if 'enPR' in iterator:
            enPR_list.append(iterator.split('enPR: ')[-1].split(', IPA')[0])
        if 'IPA' in iterator:
            ipa_list.append(iterator.split('IPA:')[-1])
    
    temp_ipa=ipa_list
    # same enPR are not distinct pronunciation
    for index in range(len(enPR_list)):
        if index!=len(enPR_list)-1:
            if enPR_list[index]==enPR_list[index+1]:
                ipa_list.remove(temp_ipa[index])
    
    # duplicate pronunciation with different ipa
    if len(ipa_list)>len(dictinfo):
        duplicate=int(len(ipa_list)/len(dictinfo))
        max_defindex*=duplicate

    if max_defindex!=-1 :
        return ipa_list[int(max_defindex)]
    else:
        for indexiter, dictinstance in enumerate(dictinfo):
            definitions=dictinstance['definitions']
            for def_unique_pos in definitions:
                posinfo=def_unique_pos['partOfSpeech']
                if pos_convert_tag(posinfo) not in targettag:
                    continue
                return ipa_list[indexiter]


#step 2: find meaning considering context
before_sort_list=[]
for sentence in sent_token:
    word_token=word_tokenize(sentence)
    heterolist=[]
    pos_heterolist=[]
    result_pos={}
    result_hetero={}
    for word in word_token:
        if word.lower() in heteronym_dict :
            heterolist.append(word.lower())
        elif word.lower() in pos_heteronym:
            pos_heterolist.append(word.lower())
            
    # non-pattern heteronym
    if len(heterolist)>0:
        hetero_unq=list(set(heterolist))
        tagged=nltk.pos_tag(word_tokenize(sentence))
        hetero_num_in_sentence=0
        hetero_num_in_sentence_noun=0
        
        for hetero_unqentry in hetero_unq:
            for i in range(len(tagged)):
                if hetero_unqentry==tagged[i][0].lower():
                    targetword=tagged[i][0].lower()
                    targettag=tagged[i][1]
                    # verb or adjective case
                    if 'VB' in targettag or 'JJ' in targettag:
                        hetero_num_in_sentence+=1
                        examplesentlist={}
                        for synset in wordnet.synsets(targetword):
                            if synset.pos()==targettag[0].lower():
                                for example in synset.examples():
                                    if targetword in example:
                                        examplesentlist[synset]=example
                        #NP chunking
                        grammar = r"""
                          NP: {<PRP\$|DT|PP\$>?<JJ>*<NN.*>}   # chunk determiner/possessive, adjectives and noun
                                {<NNP>+}                # chunk sequences of proper nouns
                        """
                        cp = nltk.RegexpParser(grammar)
                        
                        tree_flag=0
                        parsed_cmp = cp.parse(tagged)
                        iteration=0
                        for i in range(len(parsed_cmp)):
                            if targetword in parsed_cmp[i][0]:
                                iteration+=1
                                if iteration==hetero_num_in_sentence:
                                    if i+1!=len(parsed_cmp):
                                        if type(parsed_cmp[i+1])==nltk.tree.Tree:
                                            tree_flag=1

    
                        context_syn=[]
                        #chunking with example sentence
                        for examplelist in examplesentlist.values():
                            example_token=word_tokenize(examplelist)
                            pos_example=nltk.pos_tag(example_token)

                            example_tree = 0
                            parsed_exp = cp.parse(pos_example)
                            for i in range(len(parsed_exp)):
                                if targetword in parsed_exp[i][0]:
                                    if i+1!=len(parsed_exp):
                                        if type(parsed_exp[i+1])==nltk.tree.Tree:
                                            example_tree=1

                            if example_tree == tree_flag:
                                for synsetelem, example in examplesentlist.items():
                                    if example==examplelist:
                                        context_syn.append(synsetelem)
                        #meaning specified -> map meaning to pronunciation
                        classified_pronun= map_meaning2pronun(context_syn,targetword,targettag)
                        result_hetero.setdefault(targetword, [])
                        result_hetero[targetword].append(classified_pronun)

                    #noun case
                    elif 'NN' in targettag:
                        result_syn=[]
                        hetero_num_in_sentence_noun+=1
                        iteration=0
                        # pywsd used
                        wsdlist= disambiguate(sentence)
                        for wsdentry in wsdlist:
                            if wsdentry[0].lower()==targetword and wsdentry[1] is not None: 
                                if wsdentry[1].pos()=='n':
                                    iteration+=1
                                    if hetero_num_in_sentence_noun==iteration:
                                        result_syn.append(wsdentry[1])

                        classified_pronun= map_meaning2pronun(result_syn,targetword,targettag)
                        result_hetero.setdefault(targetword, [])
                        result_hetero[targetword].append(classified_pronun)

    # pattern heteronym
    if len(pos_heterolist)>0:
        pos_hetero_unq=list(set(pos_heterolist))
        tagged=nltk.pos_tag(word_tokenize(sentence))
      
        for pos_hetero_unqentry in pos_hetero_unq:
            for i in range(len(tagged)):
                if pos_hetero_unqentry==tagged[i][0].lower():
                    targetword=tagged[i][0].lower()
                    targettag=tagged[i][1]

                    if 'VB' in targettag:
                        dictate=pos_heteronym[targetword]['V']
                        result_pos.setdefault(targetword, [])
                        result_pos[targetword].append(' '.join(dictate))
                    elif 'JJ' in targettag:
                        dictate=pos_heteronym[targetword]['J']
                        result_pos.setdefault(targetword, [])
                        result_pos[targetword].append(' '.join(dictate))
                    elif 'NN' in targettag:
                        dictate=pos_heteronym[targetword]['N']
                        result_pos.setdefault(targetword, [])
                        result_pos[targetword].append(' '.join(dictate))



    if len(heterolist)+len(pos_heterolist)!=0:
        before_sort_list.append([sentence,result_hetero,result_pos])


#rank
for i in range(len(before_sort_list)):
    for j in range(i+1,len(before_sort_list)):
        stdnum1=0
        stdnum2=0
        cmpnum1=0
        cmpnum2=0
        #rule 1
        for stdkey1 in before_sort_list[i][1].keys():
            stdnum1+=len(before_sort_list[i][1][stdkey1])
        for stdkey2 in before_sort_list[i][2].keys():
            stdnum2+=len(before_sort_list[i][2][stdkey2])
        for cmpkey1 in before_sort_list[j][1].keys():
            cmpnum1+=len(before_sort_list[j][1][cmpkey1])
        for cmpkey2 in before_sort_list[j][2].keys():
            cmpnum2+=len(before_sort_list[j][2][cmpkey2])
        stdnum=stdnum1+stdnum2
        cmpnum=cmpnum1+cmpnum2
        if stdnum<cmpnum:
            before_sort_list[i],before_sort_list[j]=before_sort_list[j],before_sort_list[i]
        elif stdnum==cmpnum:
            # rule 2
            std_distinct=len(before_sort_list[i][1].keys())+len(before_sort_list[i][2].keys())
            cmp_distinct=len(before_sort_list[j][1].keys())+len(before_sort_list[j][2].keys())
            if std_distinct>cmp_distinct:
                before_sort_list[i],before_sort_list[j]=before_sort_list[j],before_sort_list[i]
            elif std_distinct==cmp_distinct:
                # rule 3 -changed! 
                if stdnum1<cmpnum1:
                    before_sort_list[i],before_sort_list[j]=before_sort_list[j],before_sort_list[i]

f=open('CS372_HW3_output_20170396.csv','w',newline="")
linecount=0     
for sortedsent in before_sort_list:
    if linecount==30:
        break
    f.write(str(sortedsent)+'  Cited from The Complete Works of William Shakespeare''\n')
    linecount+=1
