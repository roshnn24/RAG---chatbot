from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
import os

# Set environmental variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] ='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'

app = Flask(__name__)

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = 'uploads'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_file):
    raw_text = ''
    for page in pdf_file.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text


pdf_path = ""  # Variable to store the path of the uploaded PDF
texts = []  # Variable to store the text extracted from the uploaded PDF

# Load the PDF and set up models
embeddings = HuggingFaceEmbeddings()
document_search = None
chain = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global pdf_path, texts, document_search, chain
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)

        # Extract text from the uploaded PDF
        pdf_reader = PdfReader(pdf_path)
        raw_text = extract_text_from_pdf(pdf_reader)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Create document search index
        embeddings = HuggingFaceEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)

        # Load question-answering model
        chain = load_qa_chain(HuggingFaceHub(repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1'), chain_type="stuff")

        return render_template('index.html', success='File uploaded successfully')

    else:
        return render_template('index.html', error='Invalid file format')


@app.route('/get_response', methods=['POST'])
def get_response():
    if pdf_path == "":
        return render_template('index.html', error='Please upload a PDF file first')

    prompt = "summarize the code in 467 characters"  # Predefined prompt
    docs = document_search.similarity_search(prompt)
    response = chain.run(input_documents=docs, question=prompt)
    output = "".join(response)
    output = (output[output.index('Answer:'):])
    return render_template('index.html', response=output)


if __name__ == '__main__':
    app.run(debug=True)