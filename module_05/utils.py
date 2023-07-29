#####################
## imports
#####################
import json
import requests
from retry import retry
import streamlit as st

import chromadb

#####################
## Set Constants
#####################
HF_TOKEN = ''
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Constants for embedding model
EMB_MODEL_ID = 'pinecone/mpnet-retriever-discourse'
EMB_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMB_MODEL_ID}"

# Constants for QA model
QA_MODEL_ID = 'deepset/roberta-base-squad2'
QA_API_URL = f"https://api-inference.huggingface.co/models/{QA_MODEL_ID}"


#####################
## Utility Functions
#####################

def get_lines(uploaded_file):
  """
    Utility to read raw text file in binary
  """
  raw_data = []
  for line in uploaded_file:
        raw_data.append(line.decode("utf-8") )
  return raw_data



def get_embeddings(texts):
    """
      Utility to generate embeddings
    """
    response = requests.post(EMB_API_URL, headers=HEADERS, json={"inputs": texts})
    result = response.json()
    if isinstance(result, list):
      return result
    elif list(result.keys())[0] == "error":
      raise RuntimeError(
          "The API did not return a response"
          )

def load_data(db, documents):
  """
    Utility to add/index data into vector db
  """
  for i,d in enumerate(documents):
    db.add(
      documents=d,
      ids=str(i)
    )

def get_relevant_documents(query, db):
  """
    Utility to retrieve relevant documents from vector DB
  """
  relevant_doc = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return relevant_doc

def get_answer(question,context):
    """
      Utility to leverage QA model for answering question using given context
    """
    payload = {
        "question": question,
        "context":context
    }
    data = json.dumps(payload)
    response = requests.request("POST", QA_API_URL, headers=HEADERS, data=data)
    try:
      decoded_response = json.loads(response.content.decode("utf-8"))
      return decoded_response['answer'], decoded_response['score'], ""
    except Exception as ex:
      return "Apologies but I could not find any relevant answer", 0.0, ex

def create_db():
  """
    Utility to instantiate vector db client and collection
  """
  chroma_client = chromadb.Client()
  db = chroma_client.get_or_create_collection(name="nlp_llm_workshop",
                                      embedding_function=get_embeddings)
  return chroma_client,db


def sidebar():
    """
      Utility to add content to sidebar
    """
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload a txt file📄\n"
            "3. Ask a question about the document💬\n"
        )

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📖PersonalGPT is a demo to showcase retrieval augmented question answering system"
        )
        st.markdown("Made by [raghav bali](https://twitter.com/rghv_bali)")
        st.markdown("---")