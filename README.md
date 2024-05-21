# Using AI to give reccomendation for Hair Dryer companies for improvements to current product

Simple overview:
1. Use data-driven text analysis
2. Identify Design Opportunities

## Description

Our project aims to do the follwing:
    a. Scrape categories for classification
    b. Identify differences between company's hair dryer and competitors' hair dryer 
    c. Based on the differences identified earlier
    d. Use AI to reccomend changes/additions
    e. Give specifications on top of those changes
    
To which it will give the consulting hair dryer company results for improvements to their existing product based on other product reviews and general comment consensus.

The rationale for using data-driven text analysis is because through website reviews, and video transcripts and comments, it grants us ease of access to information as well as ease of obtaining multiple sources. 

## Getting Started

### Dependencies

* We used Python 3.12.2 to run the code
* We ran it in Google Collab for ease of sharing accessibility

### Initialisation

* Import libraries for use

* Installing the necessary libraries

* Initialisiing for use in Google Colab (ONLY IF USING GOOGLE COLAB)
** Do replace <path> with your relevant Google Drive path

* Initialising api keys and some helper functions
** Examples of the youtube scraping parameters and headers for google search given below
```python
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

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
}
```

* Defining some functions to use in our code
```python
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
```

### Executing program

* Allowing inputs for consulting company
```python
company_name = input('Enter your company name: ')
```
(For example's sake we will use Dyson as our consulting company)

* Parse through reputable, unbiased websites to get a list of initlal classification terms
```python
# sites to obtain keyphrases to be parsed to make catagories
links = ['https://agarolifestyle.com/blogs/blogs-listing/hair-dryer-attachments-ultimate-guide-to-use-them',
         'https://mrbarber.in/blogs/news/a-quick-guide-to-choosing-the-best-hair-dryer#:~:text=A%20good%20hair%20dryer%20should,to%20cater%20to%20your%20needs.'
         ]
```

* Run this initial list through ChatGPT-4 to get an updated list of classification terms related to hair dryers
```python
#database creation
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

prompt = question + template
result = qa_chain({"query": prompt})
```

* Reset the vectordb to be empty for reinitialisation when we run another LLM - IMPORTANT OR ELSE CODE WILL NOT CONTINUE FURTHER DOWN
```python
vectordb = None
```

* Get a web scrape of the consulting company's hair dryer and get the top search (to prevent repeats)

* Get a youtube scrape of the competition companies' hair dryers from top 5 review videos' transcripts

* Get a youtube scrape of the competition companies' hair dryers from top 5 review videos' comments for sentiment analysis

* Initialising directories for storage and future access

* Preparing the data for database processing before running the LLM

* Running through the LLm to classify the data with the classifications as obtained previously and cleaning up labels by similarity
```python
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
```

* Run the new database through a sentiment analysis LLM to get general consensus on features and adding the sentiment label to each text split
```python
model = "cardiffnlp/twitter-roberta-base-sentiment"                       # negative, neutral, positive
classifier = pipeline("sentiment-analysis", model=model)
sentiment_results = []
# results = classifier(splits)
for i in range(len(splits)):
    sentiment_results.append(classifier(splits[i].page_content[1:514])) # spliced as such as sentiment model can only take up to 514 characters of text per split
for i in range(len(sentiment_results)):
print(i)
parts[i]["sentiment_label"] = sentiment_results[i][0]["label"]
```

* Reset the vectordb for future LLM model runs
```python
vectordb = None
```

* Separating the dataset into two subgroups for ChatGPT querying 
```python
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
```
* Initilising the vectordbs for the first subgroup

* Parsing dataset 1 through ChatGPT-4 to identify design opporunities

* Initilising the vectordbs for the second subgroup

* Parsing dataset 2 through ChatGPT-4 to identify design opporunities

## Disclaimers

Still in development - AI may not give ideal nor accurate answers

## Authors

DAI 2024 Group 6
* Brendan
* Sahitya
* Jya Yin
* Cion
* Le Zhan
* Timothy
