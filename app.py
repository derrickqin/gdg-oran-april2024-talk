import streamlit as st
from langchain.vectorstores.utils import DistanceStrategy
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import BigQueryVectorSearch
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAI

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_google_vertexai import VertexAI
from langchain.chains import RetrievalQA

PROJECT_ID = "derrick-doit-sandbox"
REGION = "US"
DATASET = "vector_search"
TABLE = "doc_and_vectors"

llm = VertexAI(model_name="gemini-pro", temperature=0)

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko", project=PROJECT_ID
)

store = BigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location=REGION,
    embedding=embedding,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)

prompt_template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.

{context}

Please follow the following rules:
1. If the question is to request links, please only return the source links with no answer.
2. Don't rephrase the question. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following links** and add the source links as a list.
3. If you find the answer, write the answer in a concise way with many details and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.
4. The answer should be in following format. Keep an eye on the changeline and don't truncate the link:

**Question**: {question}
\n**Answer**:
\n**Source**:
"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(llm=VertexAI(model_name="gemini-1.0-pro", temperature=0), prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=True)
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)


def generate_response(question):
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        callbacks=None,
        retriever=store.as_retriever(),
        return_source_documents=True,
    )
    result = qa.invoke(question)
    st.info(result["result"])

st.title('🦜🔗 Retrieval Augmented Generation with Google Gemini and BigQuery')

with st.form('my_form'):
    text = st.text_area('Enter text:', "What is the deprecation date of PaLM API?")
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)