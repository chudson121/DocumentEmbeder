from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb 
# from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
# from langchain_community.embeddings import OllamaEmbeddings

import jsonpickle
import ollama #not to be confused with langchains implementation of ollama they are very different,
import time
import os
import json
from pathlib import Path
from DocumentChunk import *
import numpy as np
from numpy.linalg import norm

def get_document_types(folder_paths):
    document_types = set()
    for folder_path in folder_paths:
        for dirpath, subdirs, files in os.walk(folder_path):
            for file in files:
                file_extension = os.path.splitext(file)[1].lower()
                if(file_extension not in document_types):
                    document_types.add(file_extension)
        return document_types

def FilterFiles(fileArray, extensions):
    filteredFiles = []
    if fileArray:
        for file in fileArray:
            file_extension = os.path.splitext(file)[1].lower()
            if(file_extension in extensions):
                filteredFiles.append(file)

    if filteredFiles:
        filteredFiles.sort()        
    print("There are", len(filteredFiles), "in the filtered list")
    return filteredFiles

def list_files(folder_paths):
    file_list = []
    for folder_path in folder_paths:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_list.append(os.path.join(root, file))
        
    if file_list:
        file_list.sort()
        
    print("There are", len(file_list), "in the full file list")
    return file_list

# Problem is the chunk to embedding mapping, vector db wants clear text, embedding, and id
def Process_Text_Documents(file_array, vector_collection, embedding_model_name):
    # create dir if it doesn't exist this needs to be app level not at save
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

#TODO Make this multi threaded/pooled
    for file in file_array:
        print('Processing document:', file)
        
        start_time = time.perf_counter()
        
        documentPartsArray = get_embeddings_for_TextFiles(file, embedding_model_name)

        save_embeddings_to_file(file, documentPartsArray)
        
        # print('Adding chunks to vector database:', len(documentPartsArray))
        for embed in documentPartsArray:
            print(".", end="", flush=True) #Visual queue that its doing something for cli.
            vector_collection.add([embed.filename],[embed.embeddings], documents=[embed.text_chunks], metadatas={"source": file})

        # Calculate the end time and time taken
        end_time = time.perf_counter()
        print(f"Elapsed time: {end_time - start_time} seconds")
        print("There are", vector_collection.count(), "in the collection")


def save_embeddings_to_file(filename, DocumentChunks):
    
    # dump embeddings to json
    if(len(DocumentChunks) == 0):
        print('No embeddings to save')
        return
    
    # for doc in DocumentChunks:
    #     print(doc.filename)
    #     print(doc.embeddings)
    #     print(doc.text_chunks)
    #     print('-------------------------------------------')

    
    json_string = jsonpickle.encode(DocumentChunks)

    with open(f"embeddings/{create_safe_filename(filename)}.json", "w") as f:
        json.dump(json_string,  f)

def create_safe_filename(longfilename):
    base_path, filename = os.path.split(longfilename)
    path = Path(longfilename)
    # Go one level up from base_path
    new_filename = path.parent.name + '_' + filename

    return new_filename

def load_embeddings(filename):
    
    convert_filename_to_safe = create_safe_filename(filename)
    
    # check if file exists
    if not os.path.exists(f"embeddings/{convert_filename_to_safe}.json"):
        return False
    
    # load embeddings from json
    with open(f"embeddings/{convert_filename_to_safe}.json", "r") as f:
        recreated_obj = jsonpickle.decode(json.load(f))
        return recreated_obj

def get_embeddings_for_TextFiles(filename, modelname):
    docChunksArray = []

    start_time = time.perf_counter()
    
    # check if embeddings are already saved
    if (savedDocData := load_embeddings(filename)) is not False:
        objectArray = savedDocData
    else:
        document_data = get_file_contents(filename)
        print('splitting document:', filename)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=vectorDB_ChunkSize, 
            chunk_overlap=vectorDB_ChunkOverlap, 
            length_function=len,  
            separators=["\n\n", "\n", " ", ".", ",", "",
                        "\u200b",  # Zero-width space
                        "\uff0c",  # Fullwidth comma
                        "\u3001",  # Ideographic comma
                        "\uff0e",  # Fullwidth full stop
                        "\u3002",  # Ideographic full stop
                ])    
        chunks = text_splitter.split_text(document_data)
        print('embedding document chunks:', len(chunks))

        for index, chunk in enumerate(chunks):
            print(".", end="", flush=True)
            chunk_embedding = ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
            docChunksArray.append(DocumentChunks_With_Embeddings(chunk_embedding, chunk, filename+str(index)))
        
    print()
    print(f"Embedding Loading Time: {time.perf_counter() - start_time} seconds")
    print(f"{len(docChunksArray)} embeddings created for {filename}")

    return docChunksArray

def get_file_contents(file):
    print('Reading document:', file)
    with open(file, 'r', encoding="utf-8-sig") as f:
        return f.read()

def ChromaDBConfig(embedding_model_name):
    persist_directory = './db' #needs to come from a settings file
    vectorDBClient = chromadb.PersistentClient(path=persist_directory)
    print('HEARTBEAT:', vectorDBClient.heartbeat())
    collection = vectorDBClient.get_or_create_collection("docs") #TODO make a variable from settings
    print("There are", collection.count(), "in the collection")
    return collection   

# find cosine similarity of every chunk to a given embedding
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def QueryLLM(vector_collection, model_name, embedding_model_name, user_prompt):
    #TODO put in error handling if the model fails
    while True:
        
        if len(user_prompt) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        
        start_time = time.perf_counter()
        print(f"\nThinking using {model_name}...\n")
        
        prompt_chunk_embedding = ollama.embeddings(model=embedding_model_name, prompt=user_prompt)["embedding"]
        # prompt_embedding = ollama.embeddings(model=model_name, prompt=user_prompt)["embedding"]
<<<<<<< HEAD
        # print (f"Embedded prompt: {prompt_chunk_embedding}")
=======
        print (f"Embedded prompt: {prompt_chunk_embedding}")
>>>>>>> 77ac6fc (removed embeddings)
        results = vector_collection.query( 
            query_embeddings=[prompt_chunk_embedding],
            n_results=10
            #,include=["documents", "metadatas", 'distances']
            #,where_document={"$contains":user_prompt}  # optional filter
        )

<<<<<<< HEAD
        # print(results)
        # langchain implementation not avail on original chromadb
=======
        print(results)
        #langchain implementation not avail on original chromadb
>>>>>>> 77ac6fc (removed embeddings)
        # results2 = vector_collection.similarity_search(query_embeddings=[prompt_chunk_embedding], n_results=10, include=["documents", "metadatas"])
        # print(results2)

        data = results['documents'][0][:10] #only get first page
        # data = results['documents']
        # print(data)

        response = ollama.chat(model=model_name, 
            messages=[
                {
                    "role": "system",
                    "content": f"{System_prompt} \n {data}"
                },
                {"role": "user", "content": user_prompt},
            ],
        )
    
        print("\n\n")
        print(response["message"]["content"])
        print(f"query Time: {time.perf_counter() - start_time} seconds")
    
    
    # QA_CHAIN_PROMPT = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template=System_prompt,
    # )

    # llm = Ollama(model="llama3:8b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    # qa_chain = RetrievalQA.from_chain_type(llm, 
    #     retriever=vectorstore.as_retriever(),
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    # )

    # result = qa_chain({"query": userPrompt})
    
    # docs = db4.similarity_search(query)
    # for doc in docs:
    #     print(doc.page_content)
    #     #print(docs[0].page_content)


    #return result



#Clear console window
os.system('cls' if os.name == 'nt' else 'clear')

embedding_model_name='nomic-embed-text'
query_Model = 'dolphin-llama3' #'llama3' # 'dolphin-llama3' 'llava-phi3' 'llama3' 'falcon2'
#falcon2 is slow
# 'llava-phi3' gave what a 1/6 answer WTF????
# dolphin-llama3 64sec
# llama3 93sec

vectorDbCollection = ChromaDBConfig(embedding_model_name)
vectorDB_ChunkSize=1500 
vectorDB_ChunkOverlap=100 

folder_paths = set()
folder_paths.add('E:\Blog\RandomThoughts\Articles')

<<<<<<< HEAD
=======
#TODO add flag to run query vs process
# QueryLLM(vectorDbCollection, query_Model, embedding_model_name)

>>>>>>> 77ac6fc (removed embeddings)
allFiles = list_files(folder_paths)

#TODO need to add a filter date and only acquire files modified from that date
csvFiles = FilterFiles(allFiles, ['.csv'] )
docxFiles = FilterFiles(allFiles, ['.docx', '.doc'] )
excelFiles = FilterFiles(allFiles, ['.xls', '.xlsx'])
imageFiles = FilterFiles(allFiles, ['.png', '.jpg', '.jpeg'])
onenoteFiles = FilterFiles(allFiles, ['.one'])
pdfFiles = FilterFiles(allFiles, ['.pdf'] )
pptxFiles = FilterFiles(allFiles, ['.ppt', '.pptx'])
txtFiles = FilterFiles(allFiles, ['.txt', '.md', '.json', '.html'])


<<<<<<< HEAD
System_prompt = """you are an expert principal researcher for fortune 1000 businesses.You are always able to find and assemble useful, truthful and timely knowledge. Your audience is senior business and technology leadership at large organizations
Use an informative and persuasive tone throughout, drawing clear comparisons to simplify complex concepts. Start with a compelling headline that indicates the value of the article. Follow with an engaging introduction that outlines a relevant problem or challenge. Include why the topic is pertinent to senior leaders, using industry-specific examples. Provide clear, actionable solutions to the problem, backed by relevant data, case studies, or success stories. Incorporate insights from industry experts and use visuals like infographics, charts, or graphs to support your points. Conclude by summarizing the key points, reinforcing the value of the solutions provided, and providing a clear call-to-action. Ensure the content is concise, direct, and valuable, and easily shareable on social media platforms. Create an overall mood of empowerment and assurance, positioning the discussed solution as a viable and beneficial option for businesses. Generate your response by following the steps below:
1. Recursively break-down the post into smaller questions/directives
2. For each atomic question/directive:
2a. Select the most relevant information from the context in light of the conversation history
3. Generate a draft response using the selected information, whose brevity/detail are tailored to the posters expertise
4. Remove duplicate content from the draft response
5. Generate your final response after adjusting it to increase accuracy and relevance
6. Now only show your final response! Do not provide any explanations or details

Context: 
"""
#TODO add flag to run query vs process

#TODO move this to a query class along with dependencies
user_prompt = input("what do you want to know? -> ")
QueryLLM(vectorDbCollection, query_Model, embedding_model_name, user_prompt)

# Process_Text_Documents(txtFiles, vectorDbCollection, embedding_model_name)
=======
Process_Text_Documents(txtFiles, vectorDbCollection, embedding_model_name)
>>>>>>> 77ac6fc (removed embeddings)
print('-------------------------------------------')
# ProcessDocument(docxFiles)
# print('-------------------------------------------')
# ProcessDocument(pdfFiles)
# print('-------------------------------------------')
# ProcessDocument(pptxFiles)
# print('-------------------------------------------')

# References
# https://decoder.sh/videos/rag-from-the-ground-up-with-python-and-ollama
# https://github.com/ollama/ollama-python
# https://ollama.com/blog/embedding-models
