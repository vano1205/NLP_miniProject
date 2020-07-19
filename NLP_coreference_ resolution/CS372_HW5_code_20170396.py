import nltk
import csv
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import Tree
import bs4
import requests

snippet=open('CS372_HW5_snippet_output_20170396.tsv','w',newline="")
page=open('CS372_HW5_page_output_20170396.tsv','w',newline="")
gaptest=open('gap-test.tsv','r',newline="")

snippet_write = csv.writer(snippet)
snippet_write.writerow(['ID'+"	"+'A-coref'+"	"+'B-coref'])

page_write = csv.writer(page)
page_write.writerow(['ID'+"	"+'A-coref'+"	"+'B-coref'])

data = csv.reader(gaptest,delimiter="	")
header = next(data)

grammar = r"""
    PARA: {<PARA><FW|CD|NN.*|JJ.*|VB.*|DT|IN|CC>*<PARA>}
    NP : {(<JJ.*|DT><IN>)?<DT>?<VBG|VBN><JJ.*|NN.*|VBG>+(<CC|SYM><DT|JJ.*|FW|PARA|CD|PRP.*|NN.*>+)?} 
         {(<JJ.*|DT><IN>)?<DT|JJ.*|FW|PARA|CD|PRP.*|POS|NN.*>+(<CC|SYM><DT|JJ.*|FW|PARA|CD|PRP.*|NN.*>+)?}
    PP : {<IN><CD>?<NP>}
    VP : {<MD|VBD|VBZ>?<RB>?<VB.*>+(<CC><VB.*>)?<TO|NP|PP|RB.*>+(<CC>+<TO|NP|PP|RB.*>)?}
    CLAUSE : {<NP>+<PP>?<VP>+(<CC><VP>+)?}
    EQUAL : {<,><NP><,>}
"""

cp = nltk.RegexpParser(grammar)

# extracting subject from sentence - extract NNP from NP inside CLUASE
def extractSubject(parsedsent):
    resultlist=[]
    for parsedentry in parsedsent:
        if type(parsedentry) == Tree and parsedentry.label() == 'CLAUSE':
            for clause in parsedentry:
                if type(clause) == Tree and clause.label() == 'NP':
                    subjectlist=[]
                    for name_part in clause.leaves():
                        if 'NNP' in name_part[1] :
                            if name_part[0] not in '][-`':
                                subjectlist.append(name_part[0])
                    subject = ' '.join(subjectlist)
                    if subject!='':
                        resultlist.append(subject)
    return resultlist

#collect all NP by traversing tree
def collect_NP(parsedsent,nplist):
    for node in parsedsent:
        if type(node) is nltk.Tree:
            if node.label() == 'NP':
                for leaveentry in node.leaves():
                    nplist.append(leaveentry[0])

            collect_NP(node,nplist)
    return nplist

# check if there is a NP chunk collected from collect_NP between candidate and pronoun
def btw_NP(NP_list, candidate, pronoun):
    candidate_lastword = word_tokenize(candidate)[-1]
    candidate_firstword = word_tokenize(candidate)[0]
    for i in range(len(NP_list)):
        if NP_list[i] == pronoun:
            if i != len(NP_list)-1 and NP_list[i+1] == candidate_firstword:
                return False
            elif i != 0 and NP_list[i-1] == candidate_lastword:
                return False
    return True

def get_resolution(contextflag, testlist):
    first_gramlist = []

    candidate1_answer = -1
    candidate2_answer = -1
    close_answer = -1
    far_answer = -1

    pronoun = testlist[2]
    candidate1 = testlist[4]
    candidate2 = testlist[7]

    pronoun_offset = int(testlist[3])
    candidate1_offset = int(testlist[5])
    candidate2_offset = int(testlist[8])

    document = testlist[1]
    sent_token = sent_tokenize(document)
        
    # find pronoun containing sentence
    offset = 0
    for sentence in sent_token:
        newoffset = offset+len(sentence)+1
        if pronoun_offset >= offset and pronoun_offset < newoffset:
            target_sent = sentence
            break
        offset = newoffset
    

    word_token = word_tokenize(target_sent)
    pos_tag = nltk.pos_tag(word_token)

    thattag=('that','THAT')
    pos_tag = [thattag if thattag[0] == e[0] else e for e in pos_tag]

    # find close, far candidate in terms of distance from pronoun
    candidate1_close = abs(candidate1_offset-pronoun_offset)
    candidate2_close = abs(candidate2_offset-pronoun_offset)

    if candidate1_close < candidate2_close:
        close_candid = candidate1
        close_offset = candidate1_offset
        far_candid = candidate2
        far_offset = candidate2_offset
        cand1_close = 1
    else:
        close_candid = candidate2
        close_offset = candidate2_offset
        far_candid = candidate1
        far_offset = candidate1_offset
        cand1_close = 0

    parsedsent=cp.parse(pos_tag)

    # extract subject from pronoun containing sentence
    first_gramlist = extractSubject(parsedsent)

    # all cand1, cand2, pronoun in same sentence
    if close_offset >= offset and far_offset >= offset:
        # pronoun, candidate - which comes first?
        if close_offset > pronoun_offset:
            late_close = 1
        else:
            late_close = 0

        # first word -> look at previous sentence
        if pronoun[0].isupper():
            close_answer = False
            far_answer = False
        
        # no named entity subject
        elif len(first_gramlist)==0:
            # pronoun comes first, no named subject -> False
            if late_close == 1:
                close_answer = False
            else:
                close_answer = True
                far_answer = False
        else:
            for gram_entry in first_gramlist:
                # close candidate in subject
                if close_candid in gram_entry or gram_entry in close_candid:
                    # check if NP is in between close candidate and pronoun
                    NP_list = collect_NP(parsedsent, [])
                    close_answer = btw_NP(NP_list, close_candid, pronoun)
                    if close_answer is False and (far_candid in gram_entry or gram_entry in far_candid):
                        far_answer = True
                    elif close_answer is True:
                        far_answer = False
            
            if close_answer == -1 or far_answer == -1:
                close_answer = False
                # far candidate in subject
                for gram_entry in first_gramlist:
                    # no need btw_NP -> close candidate exist in the same sentence
                    if far_candid in gram_entry or gram_entry in far_candid:
                        far_answer = True
                if far_answer == -1:
                    far_answer = False

    # close candidate, pronoun same sentence
    if close_offset >= offset and close_answer == -1:
        if pronoun[0].isupper():
            close_answer = False

        if close_offset > pronoun_offset:
            early_pronoun = 1
        else:
            early_pronoun = 0

        if close_answer == -1:  
            # subject is not named entity 
            if len(first_gramlist) == 0:
                if early_pronoun == 1:
                    close_answer = False
                else:
                    close_answer = True
            
            else:
                # close candidate in subject
                for gram_entry in first_gramlist:
                    if close_candid in gram_entry or gram_entry in close_candid:
                        NP_list = collect_NP(parsedsent, [])
                        close_answer = btw_NP(NP_list, close_candid, pronoun)

                # another subject
                if close_answer == -1:
                    close_answer = False

        # priority to close candidate
        if close_answer is True:
            far_answer = False

    # far candidate, pronoun same sentence
    if far_offset >= offset and far_answer == -1:
        if pronoun[0].isupper():
            close_answer = False

        if far_offset > pronoun_offset:
            early_pronoun = 1
        else:
            early_pronoun = 0

        if far_answer == -1: 
            if len(first_gramlist) == 0:
                if early_pronoun == 1:
                    far_answer = False
                else:
                    far_answer = False
                
            else:
                for gram_entry in first_gramlist:
                    if far_candid in gram_entry or gram_entry in far_candid:
                        NP_list = collect_NP(parsedsent, [])
                        far_answer = btw_NP(NP_list, far_candid, pronoun)

                if far_answer == -1:
                    far_answer = False

    # look at prior sentence
    if close_answer == -1 or far_answer == -1 :
        if close_answer == -1:
            target_offset = close_offset
        else:
            target_offset = far_offset
        second_offset = 0
        
        # search for candidate containing sentence
        for sentence in sent_token:
            newoffset = second_offset+len(sentence)+1
            if target_offset >= second_offset and target_offset < newoffset:
                prev_sent = sentence
                break
            second_offset = newoffset

        word_token = word_tokenize(prev_sent)
        pos_tag = nltk.pos_tag(word_token)
        second_parsed=cp.parse(pos_tag)
        second_gram = extractSubject(second_parsed)

        # snippet-context or page-context -> nominative case pronoun with named entity subejct
        if contextflag == 0 or ((pronoun.lower() == 'she' or pronoun.lower() == 'he') and len(second_gram)!=0):

            if close_answer == -1:
                if len(second_gram)==0:
                        close_answer = True
                for gram_entry in second_gram:
                    if close_candid in gram_entry or gram_entry in close_candid: 
                        close_answer = True
                if close_answer == -1:
                    close_answer = False

            if far_answer == -1:
                if len(second_gram)==0:
                        far_answer = False
                for gram_entry in second_gram:
                    if far_candid in gram_entry or gram_entry in far_candid: 
                        far_answer = True
                if far_answer == -1:
                    far_answer =  False
        # page-context -> wikipedia page
        else:        
            wikiurl = testlist[10]
            wikitext = ''
            response = requests.get(wikiurl)

            if response is not None:
                html = bs4.BeautifulSoup(response.text, 'html.parser')

                paragraphs = html.select("p")
                for para in paragraphs:
                    wikitext += para.text
            wiki_sentlist = sent_tokenize(wikitext)

            wiki_close = 0
            wiki_far = 0
            named_subject = 0

            # analyze each sentence in wikipedia page
            for wiki_sent in wiki_sentlist:
                word_tokened = word_tokenize(wiki_sent)
                if pronoun.lower() in word_tokened:
                    wiki_tagged = nltk.pos_tag(word_tokened)
                    wiki_parsed = cp.parse(wiki_tagged)

                    wiki_gram = extractSubject(wiki_parsed)

                    if len(wiki_gram) == 0:
                        continue

                    else:
                        named_subject += 1
                        for wiki_candid in wiki_gram:    
                            if wiki_candid in close_candid or close_candid in wiki_candid:
                                wiki_close += 1  
                            if wiki_candid in far_candid or far_candid in wiki_candid:
                                wiki_far += 1
            # choose the candidate with more named entity subject instances
            if wiki_close != 0 or wiki_far != 0:
                if wiki_close >= wiki_far:
                    close_answer = True
                    far_answer = False
                else:
                    close_answer = False
                    far_answer = True

            else:
                # no named entity subject -> referring non-subject case : True
                if named_subject == 0:
                    if close_answer == -1:
                        close_answer = True

                    if far_answer == -1:
                        far_answer = True

                # there is named entity subject, but not candidate -> False reference
                else:
                    if close_answer == -1:
                        close_answer = False

                    if far_answer == -1:
                        far_answer = False

            
    if cand1_close == 1:
        candidate1_answer = close_answer
        candidate2_answer = far_answer
    else:
        candidate1_answer = far_answer
        candidate2_answer = close_answer

    if contextflag == 0:
        snippet_write.writerow([testlist[0]+"	"+str(candidate1_answer)+"	"+str(candidate2_answer)])
    else:
        page_write.writerow([testlist[0]+"	"+str(candidate1_answer)+"	"+str(candidate2_answer)])
    

for testlist in data:
    # snippet context
    get_resolution(0,testlist)
    # page context
    get_resolution(1,testlist)     


snippet.close()
page.close()