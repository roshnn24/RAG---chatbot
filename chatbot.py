from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
import os

'''os.environ["HUGGINGFACEHUB_API_TOKEN"] ='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'
os.environ["OPENAI_API_KEY"] = "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"'''
os.environ["HUGGINGFACEHUB_API_TOKEN"] ='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'

pdfreader = PdfReader("/Users/rosh/Downloads/langy.pdf")


from typing_extensions import Concatenate
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content
# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 2000,
    chunk_overlap  = 1000,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)
chain = load_qa_chain(HuggingFaceHub(repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1'), chain_type="stuff")  # Remove max_length argument

#chain = load_qa_chain(OpenAI(), chain_type="stuff")  # Remove max_length argument
def get_Chat_response(chat):
    query = chat
    docs = document_search.similarity_search(query)
    a=chain.run(input_documents=docs, question=query)
    output = "".join(a)
    output= (output[output.index('Answer:'):])
    return output


print(get_Chat_response("Display all the API calls present in the code"))
