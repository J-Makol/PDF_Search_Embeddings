import time
t2 = time.time()
from PyPDF2 import PdfReader
from itertools import islice
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
#from sklearn.neighbors import NearestNeighbors
import numpy as np
import nltk

t3 = time.time()
model_time = t3 - t2
print('Time to load Transformer Model',model_time)


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
'''
nltk.download('punkt')
'''


#Take input query and build embedding
input = input('Ask me a question about the HMC7885 RF Amplifiers\n')
query = model.encode(input)


#Normalize query
query = query / np.sqrt((query**2).sum())



# Extracting Text from PDF
file_path = "HMC7885.pdf"

'''
#Extract whole pdf
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text


# Extract text from the PDF 
text = extract_text_from_pdf(file_path)
'''

#Extract a select few pages
page_content=""                
number_of_pages = 4

with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        for page_number in range(number_of_pages):
            #page = pdf.pages(page_number)
            page_content += pdf.pages[page_number].extract_text() 


'''
#Test File for embeddings
file = open("Example_Text_For_Chunking.txt", "r")
text = file.read()
file.close()
'''


# Splitting Text into Tokens using NLTK
def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

sentence = split_text_into_sentences(page_content)


'''
# Splitting Text into Tokens using LangChain 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    length_function = len,
    separators = ['\n']
)

sentence = text_splitter.create_documents(text)
print(f'### Chunk 1: \n\n{sentence[0].page_content}\n\n=====\n')
print(f'### Chunk 2: \n\n{sentence[1].page_content}\n\n=====')
'''



'''
#Test sentences to build embeddings
sentence = ['The power output is 45 dBm',
            'The input power is 48V',
            'The chip is powered up in 2 stages']

'''

#Build embedding vector space
embedding = model.encode(sentence)
#print(embedding)
#print(embedding.shape)

#Normalize Embeddings
embedding = embedding / np.sqrt((embedding**2).sum(1, keepdims=True))
#print(embedding)

'''
#reshape embedding
embedding_reshape = embedding.reshape(-1,1)
#print(embedding_reshape)
'''


#Run K Nearest Neighbor
t0 = time.time()

# Calculate Dot Product between the query and all data items
similarities = embedding.dot(query)

# Sort results
sorted_ix = np.argsort(-similarities)

#Return top sentence
i=0
while i<1:
    print(sentence[sorted_ix[i]])
    i=i+1

t1 = time.time()

'''
#calculate search runtime
total = t1-t0
print(f"Runtime for dim={embedding.shape[0]}, documents_n={embedding.shape[1]}: {np.round(total,3)} seconds")
'''

'''
#print("Top 5 results:")
for k in sorted_ix[:2]:
    print(f"Point: {k}, Similarity: {similarities[k]}")
'''