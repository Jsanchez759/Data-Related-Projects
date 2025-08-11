from utils.youtube_api import get_transcript, process
from utils.model import chunking, set_up_model, set_up_embeddings, set_up_vision_model
from utils.vector_search import create_faiss_index, retrieve
from utils.chains import create_qa_chain, create_summary_chain
from utils.speech_recognition import get_speech_model, get_transcript_speech
from utils.ddg_search import get_results, scrape_and_combine

class Processor:
    def __init__(self):
        self._llm = None
        self._vlm = None
        self._embedding_model = None
        self._qa_chain = None
        self._summary_chain = None

        self.current_url = None
        self.processed_transcript = None
        self.faiss_index = None
        self.chunks = None
        self.current_audio = None
        self.audio_transcript = None
        self.audio_faiss_index = None
        self._speech_model = None

        self.current_query = None
        self.search_context = None
        self.search_faiss_index = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = set_up_model()
        return self._llm

    @property
    def vlm(self):
        if self._vlm is None:
            self._vlm = set_up_vision_model()
        return self._vlm

    @property
    def speech_model(self):
        if self._speech_model is None:
            self._speech_model = get_speech_model()
        return self._speech_model

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = set_up_embeddings()
        return self._embedding_model

    @property
    def qa_chain(self):
        if self._qa_chain is None:
            self._qa_chain = create_qa_chain(self.llm)
        return self._qa_chain

    @property
    def summary_chain(self):
        if self._summary_chain is None:
            self._summary_chain = create_summary_chain(self.llm)
        return self._summary_chain

    def process_video(self, video_url):
        if not video_url:
            return False, "Please provide a valid YouTube URL."

        if self.current_url == video_url and self.processed_transcript:
            return True, "Transcript already processed."

        transcript = get_transcript(video_url)
        if not transcript:
            return False, "Not able to get a transcript"

        self.processed_transcript = process(transcript)
        self.chunks = chunking(self.processed_transcript)
        self.faiss_index = create_faiss_index(self.chunks, self.embedding_model)
        self.current_url = video_url

        return True, "Transcript processed successfully."

    def get_summary(self, video_url):
        success, message = self.process_video(video_url)
        if not success:
            return message

        return self.summary_chain.invoke({"transcript": self.processed_transcript})

    def answer_question(self, video_url, question):
        if not question:
            return "Please provide a valid question."

        success, message = self.process_video(video_url)
        if not success:
            return message

        relevant_context = retrieve(question, self.faiss_index, k=5)
        return self.qa_chain.invoke({"context": relevant_context, "question": question})

    def process_audio(self, audio_input):
        if not audio_input:
            return False, "Please provide an audio file."

        if self.current_audio == audio_input and self.audio_transcript:
            return True, "Audio already processed."

        try:
            transcript = get_transcript_speech(self.speech_model, audio_input)
            if not transcript:
                return False, "Not able to get a transcript from audio."

            self.audio_transcript = transcript
            chunks = chunking(transcript)
            self.audio_faiss_index = create_faiss_index(chunks, self.embedding_model)
            self.current_audio = audio_input

            return True, "Audio processed successfully."
        except Exception as e:
            return False, f"Error processing audio: {str(e)}"

    def get_audio_transcript(self, audio_input):
        success, message = self.process_audio(audio_input)
        if not success:
            return message
        return self.audio_transcript

    def answer_speech_question(self, audio_input, question):
        if not question:
            return "Please provide a valid question."

        success, message = self.process_audio(audio_input)
        if not success:
            return message

        relevant_context = retrieve(question, self.audio_faiss_index, k=5)
        return self.qa_chain.invoke({"context": relevant_context, "question": question})

    def process_search(self, query):
        if not query:
            return False, "Please provide a valid search query."

        if self.current_query == query and self.search_context:
            return True, "Search already processed."

        try:
            results = get_results(query)
            if not results:
                return False, "No search results found."

            self.search_context = scrape_and_combine(results)
            chunks = chunking(self.search_context)
            self.search_faiss_index = create_faiss_index(chunks, self.embedding_model)
            self.current_query = query

            return True, "Search processed successfully."
        except Exception as e:
            return False, f"Error processing search: {str(e)}"

    def answer_search_question(self, query, question):
        if not question:
            return "Please provide a valid question."

        success, message = self.process_search(query)
        if not success:
            return message

        relevant_context = retrieve(question, self.search_faiss_index, k=5)
        return self.qa_chain.invoke({"context": relevant_context, "question": question})
