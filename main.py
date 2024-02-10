import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=os.environ["api_key"],temperature =0)


#create embeddings and vector store


embeddings =  HuggingFaceEmbeddings()
vectordb_file_path="faiss_index"

def create_vector_db():
    #load documents

    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()

    vector_db =FAISS.from_documents(documents=docs, embedding=embeddings)
    vector_db.save_local(vectordb_file_path)

# retriever  = vector_db.as_retriever()
# rdocs=retriever.get_relevant_documents("what do you understand by word 'GAN'?")

def get_qa_chain():
    vector_db = FAISS.load_local(vectordb_file_path,embeddings)
    retriever = vector_db.as_retriever(score_threshold =0.7)
    
    prompt_template =""" Given the following context and question, generate an answer based on this context.In the answer try to provide as much as text
    possible from "response" section in the source document. If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    CONTEXT : {context}

    QUESTION : {question}

    """

    PROMPT = PromptTemplate(
        template = prompt_template,
        input_variables = ["context","question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                chain_type = "stuff",
                retriever = retriever,
                input_key = "query",
                return_source_documents = True,
                                        chain_type_kwargs = {'prompt':PROMPT}

                                        )
    return chain
if __name__ == "__main__":
    # vector_store =create_vector_db()
    chain = get_qa_chain()
    print(chain("what is the differnce between GAN and discriminator?"))
    
    