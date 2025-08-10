import gradio as gr
from utils.youtube_api import get_transcript, process
from utils.model import chunking, set_up_model, set_up_embeddings
from utils.vector_search import create_faiss_index, retrieve
from utils.chains import create_qa_chain, create_summary_chain

class VideoProcessor:
    def __init__(self):
        self._llm = None
        self._embedding_model = None
        self._qa_chain = None
        self._summary_chain = None
        self.current_url = None
        self.processed_transcript = None
        self.faiss_index = None
        self.chunks = None
    
    @property
    def llm(self):
        if self._llm is None:
            self._llm = set_up_model()
        return self._llm
    
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
            return False, 'Not able to get a transcript'
        
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

processor = VideoProcessor()

with gr.Blocks(title="YouTube Video Summarizer & Q&A") as interface:
    gr.Markdown("# YouTube Video Summarizer & Q&A")
    
    video_url = gr.Textbox(
        label="YouTube Video URL", 
        placeholder="Enter the YouTube Video URL"
    )
    
    with gr.Row():
        with gr.Column():
            summarize_btn = gr.Button("Summarize Video", variant="primary")
            summary_output = gr.Textbox(label="Video Summary", lines=8)
        
        with gr.Column():
            question_input = gr.Textbox(
                label="Ask a Question About the Video", 
                placeholder="Ask your question"
            )
            question_btn = gr.Button("Ask Question", variant="secondary")
            answer_output = gr.Textbox(label="Answer", lines=8)
    
    summarize_btn.click(
        processor.get_summary, 
        inputs=video_url, 
        outputs=summary_output
    )
    
    question_btn.click(
        processor.answer_question, 
        inputs=[video_url, question_input], 
        outputs=answer_output
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)