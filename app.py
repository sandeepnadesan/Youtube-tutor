import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  # << NEW
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled, 
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript  # << Fix typo
)
from dotenv import load_dotenv
load_dotenv()

# Set API keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def get_youtube_transcript(url):
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(["en"])
        transcript_data = transcript.fetch()
        text = " ".join([item.text for item in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except VideoUnavailable:
        st.error("This video is unavailable.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region.")
    except Exception as e:
        st.error(f"Unexpected error getting transcript: {e}")
    return ""

def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

st.title("AI-Powered Tutor (Groq Edition)")
st.write("Ask questions from YouTube lecture transcripts using Groq API.")

video_url = st.text_input("Enter YouTube Video URL")

if st.button("Process Video"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            
            # Use Groq LLM
            llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")# You can also use llama3-70b
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("Transcript processed successfully! You can now ask questions.")
    else:
        st.warning("Please enter a valid YouTube URL.")

if "qa_chain" in st.session_state:
    user_question = st.text_input("Ask a question from the video transcript")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.write("**Answer:**", answer)
