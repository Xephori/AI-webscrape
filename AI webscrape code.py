import os
import shutil
from bs4 import BeautifulSoup
import requests, json, lxml
import re
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import everygrams
from transformers import pipeline
from googleapiclient.discovery import build
import pickle
import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

! pip install langchain
! pip install langchain-openai
! pip install deep-translator
! pip install google-api-python-client
! pip install googlesearch-python
! pip install youtube-transcript-api
! pip install langchain==0.1.9 chromadb==0.4.24 langchain-openai==0.0.8
nltk.download('stopwords')
stopwords = stopwords.words('english')      # set as English

from google.colab import drive
drive.mount('/content/drive')
%cd <your/Drive/path/here>

f = open("<your openai key path here>", "r")
openaikey = f.readlines()[0]
f.close()
os.environ["OPENAI_API_KEY"] = openaikey            # LangChain requires API key in environment

f = open("<your google api key path here>", "r")
googlekey = f.readlines()[0]
f.close()
youtube = build('youtube', 'v3', developerKey=googlekey)
vid_id = []             	  # video id
vid_page = []       		    # video links (https...)
vid_title = []              # video title
num_comments = []           # official number of comments
load_error = 0              # error counter
can_load_title = []         # temp. list for storing title w/o loading error
can_load_page = []          # temp. list for storing links w/o loading error
num_page = []               # comment_response page number
page_title = []             # comment_response video title
comment_resp = []           # comment_response
comment_list = []           # temp. list for storing comments
comment_data = []           # comments & replies from comment_response
all_count = 0               # total number of comments

params = {
    "q": search_terms, 
    "hl": "en",          # language
    "gl": "uk",          # country of the search, UK -> United Kingdom
    "start": 0,          # number page by default up to 0
    #"num": 100          # parameter defines the maximum number of results to return.
}
# initialise headers that allow Google to whitelist the instance to do a web search
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
}

class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])
model = 'ml6team/keyphrase-extraction-kbir-inspec'
extractor = KeyphraseExtractionPipeline(model=model)

def googleSearch(query):
    with requests.session() as c:
        url = 'https://www.google.com/search'
        query = {'q': query}
        urllink = requests.get(url, params=query)
        return urllink.url

def clean_text(text):                               # user defined function for cleaning text
    text = text.lower()                             # all lower case
    text = re.sub(r'\[.*?\]', ' ', text)            # remove text within [ ] (' ' instead of '')
    text = re.sub(r'\<.*?\>', ' ', text)            # remove text within < > (' ' instead of '')
    text = re.sub(r'http\S+', ' ', text)            # remove website ref http
    text = re.sub(r'www\S+', ' ', text)             # remove website ref www
    text = text.replace('€', 'euros')               # replace special character with words
    text = text.replace('£', 'gbp')                 # replace special character with words
    text = text.replace('$', 'dollar')              # replace special character with words
    text = text.replace('%', 'percent')             # replace special character with words
    text = text.replace('\n', ' ')                  # remove \n in text that has it
    text = text.replace('\'', '’')                  # standardise apostrophe
    text = text.replace('&#39;', '’')               # standardise apostrophe
    text = text.replace('’d', ' would')             # remove ’ (for would, should? could? had + PP?)
    text = text.replace('’s', ' is')                # remove ’ (for is, John's + N?)
    text = text.replace('’re', ' are')              # remove ’ (for are)
    text = text.replace('’ll', ' will')             # remove ’ (for will)
    text = text.replace('’ve', ' have')             # remove ’ (for have)
    text = text.replace('’m', ' am')                # remove ’ (for am)
    text = text.replace('can’t', 'can not')         # remove ’ (for can't)
    text = text.replace('won’t', 'will not')        # remove ’ (for won't)
    text = text.replace('n’t', ' not')              # remove ’ (for don't, doesn't)
    text = text.replace('’', ' ')                   # remove apostrophe (in general)
    text = text.replace('&quot;', ' ')              # remove quotation sign (in general)
    text = text.replace('cant', 'can not')          # typo 'can't' (note that cant is a proper word)
    text = text.replace('dont', 'do not')           # typo 'don't'
    text = re.sub(r'[^a-zA-Z0-9]', r' ', text)      # only alphanumeric left
    text = text.replace("   ", ' ')                 # remove triple empty space
    text = text.replace("  ", ' ')                  # remove double empty space
    return text

def combine_text(list_of_text):                     # define combine_text to take (list_of_text)
    combined_text = ' '.join(list_of_text)          # do this
    return combined_text                            # and give combined_text back

company_name = input('Enter your company name: ')

# (For example's sake we will use Dyson as our consulting company)

# sites to obtain keyphrases to be parsed to make catagories
links = ['https://agarolifestyle.com/blogs/blogs-listing/hair-dryer-attachments-ultimate-guide-to-use-them',
         'https://mrbarber.in/blogs/news/a-quick-guide-to-choosing-the-best-hair-dryer#:~:text=A%20good%20hair%20dryer%20should,to%20cater%20to%20your%20needs.'
         ]
for URL in links:
  r = requests.get(URL, headers=headers)
  soup = BeautifulSoup(r.content, 'html.parser')
  baseline = []   # making empty list to store data
  te = soup.find("main")
  if type(te) != None:
    text = soup.find_all("p")   # find the subsection from html id "main" find all the subsubsection that has the <p> heading -> means text blocks and saves it as a list
    for i in range(len(text)):  # for each index in the list "text"
        text[i] = re.sub(r'\<.*?\>', '', str(text[i]))  # identifies the pattern that has those symbols, replaces them with an empty slot, then return the rest of the string text as it is
        text[i] = re.sub(r'\&.*?\;', '', str(text[i]))  # identifies the pattern that has those symbols, replaces them with an empty slot, then return the rest of the string text as it is
        text[i] = text[i].replace('\n', '') # replace any new lines with blank slot
        baseline.append(text[i])    # add this to the list
    with open('<path>/parts.txt', 'w', newline='') as f:  # making the file
        for i in baseline:
            f.write(i + "\n")
    base = open('<path>/parts.txt', 'r+')
    basecase = base.read()
    keyphrases = extractor(basecase)
    with open('<path>/classification.txt', 'a', newline='') as f:  # making the file
        for i in keyphrases:
            f.write(str(i) + "\n")
classifi = open('<path>/classification.txt', 'r+')
classification = classifi.read()

search_terms = 'classification'
try:                                              # Create directory named after search terms, just to make sure it already exists, and creates one if it does not
    os.makedirs("<path>/%s" % search_terms)
    print("Directory", search_terms, "created")
except FileExistsError:
    print("Directory", search_terms, "exists")

#making pickle files
pickle.dump(classification, open("<path>/classification.pkl", "wb"))
# w: This indicates that the file is being opened for writing. If the file does not exist, it will be created. If the file does exist, it will be truncated (i.e., cleared) before writing.
# b: This indicates that the file is being opened in binary mode. Binary mode is used when dealing with non-text files, such as images, audio, or serialized Python objects (like when using pickle). In binary mode, no newline translations are performed, and data is written and read in binary format.

#sentence splitting
loader = TextLoader("<path>/classification.txt")
document = loader.load()
text_splitter = CharacterTextSplitter( #splits by seperator terms '\n'
    separator='\n',
    length_function=len,
    is_separator_regex=False)
splits = text_splitter.split_documents(document) #creates splits for later initializing vector database
#database creation

embedding = OpenAIEmbeddings()
#reinitialise folder
try:
    shutil.rmtree('<path>/%s/persist' % search_terms)       # remove old version
except:
    pass
persist_directory = '<path>/%s/persist' % search_terms     # create new version

vectordb = Chroma.from_documents(
    documents=splits,                           # target the splits created from the documents loaded
    embedding=embedding,                        # use the OpenAI embedding specified
    persist_directory=persist_directory         # store in the persist directory for future use
)
vectordb.persist()                              # store vectordb

llm = ChatOpenAI(model_name="gpt-4", temperature=0)               # gpt model can be changed

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
    )

question = "Based on the data given, return me a python list of keywords that are relevant to manufacturers designing a hair dryer, do not give me multiple words with the same meaning."  # input question
template = "If you don't know the answer, strictly state 'I don't know', instead of making up an answer. Keep the answer as concise as possible, with a maximum of three sentences."

prompt = question + template
result = qa_chain({"query": prompt})

# Extracting the list using regex
c = re.search(r'\[.*?\]', result['result']) #from import re #initialise c object as the
ci = c.group(0)
ci = re.sub(r'\[', '', ci)
ci = re.sub(r'\s\"', '', ci) #\s is spacebar, between each symbol/punctuation, put \, tells python that " is just a string, no functions
ci = re.sub(r'\]', '', ci)
ci = re.sub(r'\"', '', ci)
ci = re.sub(r'\'', '', ci)
f = ci.split(",")
catagories = []
for i in range(len(f)):
    catagories.append(f[i])

vectordb = None

#get my website hairdryer
search_terms = company_name + " hair dryer"
query = googleSearch(search_terms)
print(query)
r = requests.get(query)
soup = BeautifulSoup(r.content, 'html.parser')
searchWrapper = soup.find('a') #this line may change in future based on google's web page structure - this is to find the search links

page_limit = 1         # page limit, if you do not need to parse all pages
page_num = 0

data = []
# get the individial web links from the Google search results
while True:
    page_num += 1
    print(f"page: {page_num}")

    html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
    soup = BeautifulSoup(html.text, 'lxml')

    for result in soup.select(".tF2Cxc"):     # https://tedboy.github.io/bs4_doc/6_searching_the_tree.html -> expl of what .select() does -> a filter
        title = result.select_one(".DKV0Md").text
        try:
           snippet = result.select_one(".lEBKkf span").text
        except:
           snippet = None
        links = result.select_one(".yuRUbf a")["href"]

        data.append({
          "title": title,
          "snippet": snippet,
          "links": links
        })

    if page_num >= page_limit:
        break
    if soup.select_one(".d6cvqb a[id=pnnext]"): #to press next page button - this is obselete as it stands since Google removed their next page function for search - may be helpful for other Google sites
        params["start"] += 1
    else:
        break

#scrape my consultant top website
query = data[0]['links'] #initialise first query
mysite = []   # making empty list to store data
r = requests.get(query, headers=headers) #entering the link
soup = BeautifulSoup(r.content, 'html.parser')
text = soup.find_all("p", class_="typography-body")   # find the subsection from html id "main" find all the subsubsection that has the <p> heading -> means text blocks and saves it as a list
for i in range(len(text)):  # for each index in the list "text"
    text[i] = re.sub(r'\<.*?\>', '', str(text[i]))  # identifies the pattern that has those symbols, replaces them with an empty slot, then return the rest of the string text as it is
    text[i] = re.sub(r'\&.*?\;', '', str(text[i]))  # identifies the pattern that has those symbols, replaces them with an empty slot, then return the rest of the string text as it is
    text[i] = text[i].replace('\n', '') # replace any new lines with blank slot
    mysite.append(text[i])    # add this to the list
    filename = '<path>/mysite.txt'    # initialise csv file name
    with open('<path>/mysite.txt', 'w', newline='') as f:  # making the file
      for i in mysite:
          f.write(i)
          f.write('\n')
    f.close()

#scrape yt top 5 reviews
search_terms = "Top 5 hair dryers in {0}".format(datetime.date.today().year-1)
max_result = 10
request = youtube.search().list(
    q=search_terms,
    maxResults=max_result,
    part="id",
    type="video"
    )
single_vid_full_captions_list = [] #individual video's full transcript
list_of_video_captions = [] #(list of lists): list of individual video's full transcripts
search_response = request.execute()
for i in range(max_result):
    caption_line = [] #appending transcript line by line
    videoId = search_response['items'][i]['id']['videoId']
    vid_id.append(videoId)                          # a list of Video IDs
    page = "https://www.youtube.com/watch?v=" + videoId
    vid_page.append(page)                           # a list of Video links
    caption_dict = YouTubeTranscriptApi.get_transcript(videoId, languages=['en'])
    for j in caption_dict:
        caption_line.append(j['text'])
    single_vid_full_captions_list.append(caption_line)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False)

count = 0
# cleaning up the comments to make it a single text block
for l in single_vid_full_captions_list:
    count += 1
    o = ' '.join(l).replace("\n","")
    o = text_splitter.split_text(o)[:2]
    for i in o:
      list_of_video_captions.append(i)
with open('<path>/list_of_video_captions.txt', 'w', newline='') as f:  # making the file
  for h in list_of_video_captions:
      f.write(h)
      f.write('\n')
competitors = open('<path>/list_of_video_captions.txt', 'r+')

for i in range(len(vid_id)): #for each video
    try:                                        # use try/except as some "comments are turned off"
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=vid_id[i]
            )
        comment_response = request.execute()
        print(comment_response)

        comment_resp.append(comment_response)   # append 1 page of comment_response
        pages = 1
        num_page.append(pages)                  # append page number of comment_response
        page_title.append(vid_title[i])         # append video title along with the comment_response

        can_load_page.append(vid_page[i])       # drop link if it can't load (have at least 1 comment page)
        can_load_title.append(vid_title[i])     # drop title if it can't load (have at least 1 comment page)

        test = comment_response.get('nextPageToken', 'nil')         # check for nextPageToken
        while test != 'nil':                                        # keep running until last comment page
            next_page_ = comment_response.get('nextPageToken')
            request = youtube.commentThreads().list(
                part="snippet,replies",
                pageToken=next_page_,
                videoId=vid_id[i]
                )
            comment_response = request.execute()
            comment_resp.append(comment_response)                   # append next page of comment_response
            pages += 1
            num_page.append(pages)                                  # append page number of comment_response
            page_title.append(vid_title[i])                         # append video title along with the comment_response
            test = comment_response.get('nextPageToken', 'nil')     # check for nextPageToken (while loop)
    except:
        load_error += 1
vid_page = can_load_page                    # update vid_page with those with no load error
vid_title = can_load_title                  # update vid_title with those with no load error
for i in range(len(vid_title)):
    if vid_title[i] == 'YouTube':           # default error title is 'YouTube'
        vid_title[i] = 'Video_' + str(i+1)  # replace 'YouTube' with Video_1 format
for k in range(len(comment_resp)):
    count = 0                                                     # comment counter
    comments_found = comment_resp[k]['pageInfo']['totalResults']  # comments on 1 comment_response page
    count = count + comments_found
    for i in range(comments_found):
        try:
            comment_list.append(comment_resp[k]['items'][i]['snippet']['topLevelComment']['snippet']['textDisplay'])
            print(comment_resp[k]['items'][i]['snippet']['topLevelComment']['snippet']['textDisplay'])
        except:
            print("missing comment")                              # or too many comments (e.g. 7.3K comments)

try:                                              # Create directory named after search terms
    os.makedirs("<path>/%s" % search_terms)
    print("Directory", search_terms, "created")
except FileExistsError:
    print("Directory", search_terms, "exists")

try:                                              # Create directory to store current search terms
    os.makedirs("<path>")
    print("Directory <path> created")
except FileExistsError:
    print("Directory <path> exists")

f = open("<path>/%s/comments.txt" % search_terms, "w+")
for i in range(len(comment_list)):
    f.write("<<<" + comment_list[i] + ">>>")
f.close()

pickle.dump(search_terms, open("<path>/%s/searchTerms.pkl" % search_terms, "wb"))
pickle.dump(comment_list, open("<path>/%s/comment_list.pkl" % search_terms, "wb"))
pickle.dump(vid_title, open("<path>/%s/vid_title.pkl" % search_terms, "wb"))
pickle.dump(vid_page, open("<path>/%s/vid_page.pkl" % search_terms, "wb"))
pickle.dump(vid_id, open("<path>/%s/vid_id.pkl" % search_terms, "wb"))

source = "<path>/%s/comments.txt" % search_terms
destination = "<path>/comments.txt"
shutil.copyfile(source, destination)

pickle.dump(search_terms, open("<path>/searchTerms.pkl", "wb"))

for i in range(len(comment_list)):        # translate all
    try:
        comment_list[i] = GoogleTranslator(source='auto', target='en').translate(str(comment_list[i]))
    except:
        print("Exceeded 5000 characters.")
# clean text only for comments
for i in range(len(comment_list)):
    if comment_list[i] is not None:
      comment_list[i] = clean_text(comment_list[i])   # overwrite with clean_text function
    else:
      comment_list[i] = ""

all_comments = combine_text(comment_list)
all_comments = " ".join(word for word in all_comments.split() if word not in stopwords)

#making pickle files
pickle.dump(search_terms, open("<path>/%s/searchTerms.pkl" % search_terms, "wb"))
pickle.dump(comment_list, open("<path>/%s/comments.pkl" % search_terms, "wb"))

pickle.dump(search_terms, open("<path>/searchTerms.pkl", "wb"))

# Creating a list of filenames
filenames = ['<path>/mysite.txt','<path>/list_of_video_captions.txt', '<path>/comments.txt']
# Open file3 in write mode
with open('<path>/features.txt', 'w') as outfile:
    # Iterate through list
    for names in filenames:
        # Open each file in read mode
        with open(names) as infile:
            # read the data from file1 and
            # file2 and write it in file3
            outfile.write(infile.read())
        # Add '\n' to enter data of file2
        # from next line
        outfile.write("\n")
with open('<path>/features.txt', 'r') as outfile:
    # Read the contents of the file
    outfile_contents = outfile.read()
    pickle.dump(outfile_contents, open("<path>/features.pkl", "wb"))

#sentence splitting
loader = TextLoader("<path>/features.txt")
document = loader.load()
text_splitter = CharacterTextSplitter( #splits based on seperator variable
    separator='\n',
    length_function=len,
    is_separator_regex=False
)
splits = text_splitter.split_documents(document)
#reinitialise folder
try:
    shutil.rmtree('<path>/%s/persist' % search_terms)       # remove old version
except:
    pass
persist_directory = '<path>/%s/persist' % search_terms     # create new version

features = pd.read_pickle("<path>/features.pkl")
candidates = catagories                # replace the candidates to suit your needs
model = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=model)
results = []
for i in range(len(splits)):
  results.append(classifier(splits[i].page_content, candidate_labels=candidates, multi_label=True))
# reducing labels by percentage matching
parts = []
l = 0
for i in range(len(results)):
  while (l < (len(results[i]['scores']))):
        if (results[i]['scores'][l] < 0.8):
            results[i]['scores'].remove(results[i]['scores'][l])
            results[i]['labels'].remove(results[i]['labels'][l])
            l -= 1
        l += 1
  parts.append(results[i])          # results[i]['labels'] -> give the classification label for that comment

model = "cardiffnlp/twitter-roberta-base-sentiment"                       # negative, neutral, positive
classifier = pipeline("sentiment-analysis", model=model)
sentiment_results = []
# results = classifier(splits)
for i in range(len(splits)):
    sentiment_results.append(classifier(splits[i].page_content[1:514])) # spliced as such as sentiment model can only take up to 514 characters of text per split
for i in range(len(sentiment_results)):
print(i)
parts[i]["sentiment_label"] = sentiment_results[i][0]["label"]

vectordb = None

subgroup1_parts = []  #i have and bad
subgroup2_parts = [] #i dont have and good
for c in catagories: #list of classes
  for i in parts:# i is dictionary of 1x split
    if c in i["labels"]: #have
      if i["sentiment_label"] == "LABEL_0" and i["sequence"] not in subgroup1_parts:#bad, no duplicates
        subgroup1_parts.append(i["sequence"])
    elif c not in i["labels"]: #i dont have
      if i["sentiment_label"] == "LABEL_2" and i["sequence"] not in subgroup2_parts: #and good, no duplicates
        subgroup2_parts.append(i["sequence"])

f = open("<path>/subgroup1_parts.txt", "w+")
for i in range(len(subgroup1_parts)):
    f.write(subgroup1_parts[i])
f.close()

f = open("<path>/subgroup2_parts.txt", "w+")
for i in range(len(subgroup2_parts)):
    f.write(subgroup2_parts[i])
f.close()

loader = TextLoader("<path>/subgroup1_parts.txt")
document = loader.load()
text_splitter = CharacterTextSplitter( #splits based on seperator variable
    separator='\n',
    length_function=len,
    is_separator_regex=False
)
splits = text_splitter.split_documents(document)
#database creation
embedding = OpenAIEmbeddings()
if subgroup1_parts != []:
  vectordb_subgroup1_parts = Chroma.from_documents(
      documents=splits,                           # target the splits created from the documents loaded
      embedding=embedding,                        # use the OpenAI embedding specified
      persist_directory=persist_directory         # store in the persist directory for future use
  )
  vectordb_subgroup1_parts.persist()                              # store vectordb

llm = ChatOpenAI(model_name="gpt-4", temperature=0)               # gpt model can be changed
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
    )
question = "What features in {} hairdryers pale in comparison to other brands? Give me in detail and specifications of the part, as well as the justification for it.".format(company_name)  # input question
template = " If you don't know the answer, strictly state 'I don't know', instead of making up an answer. Keep the answer as concise as possible, with a maximum of three sentences."

prompt = question + template
result = qa_chain({"query": prompt})

with open('<path>/improvement.txt', 'w', newline='') as f:
    f.write(result['result'])

loader = TextLoader("<path>/subgroup2_parts.txt")
document = loader.load()
text_splitter = CharacterTextSplitter( #splits based on seperator variable
    separator='\n',
    length_function=len,
    is_separator_regex=False
)
splits = text_splitter.split_documents(document)

#database creation
embedding = OpenAIEmbeddings()
if subgroup2_parts != []:
  vectordb_subgroup2_parts = Chroma.from_documents(
      documents=splits,                           # target the splits created from the documents loaded
      embedding=embedding,                        # use the OpenAI embedding specified
      persist_directory=persist_directory         # store in the persist directory for future use
  )
  vectordb_subgroup2_parts.persist()                              # store vectordb

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb_subgroup2_parts.as_retriever(),
    return_source_documents=True
    )
question = "List me 3 well received hairdryer features that other brands have that {} do not? Answer me in detail and specifications of the part, as well as the justification for it.".format(company_name) # input question
template = " If you don't know the answer, strictly state 'I don't know', instead of making up an answer. Keep the answer as concise as possible, with a maximum of three sentences."

prompt = question + template
result = qa_chain({"query": prompt})

with open('<path>/addition.txt', 'w', newline='') as f:
  f.write(result['result'])
