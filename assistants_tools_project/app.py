import gradio as gr
from utils.processor import Processor

processor = Processor()

with gr.Blocks(title="Assitant tools") as interface:
    gr.Markdown("# Assitant tools")

    with gr.Tabs():
        with gr.TabItem("YouTube Analysis"):
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

        with gr.TabItem("Speech Analysis"):
            audio_input = gr.Audio(
                label="Upload Audio File",
                type='filepath'
            )

            with gr.Row():
                with gr.Column():
                    transcript_btn = gr.Button("Show Transcript", variant="secondary")
                    transcript_output = gr.Textbox(label="Audio Transcript", lines=8)

                with gr.Column():
                    speech_question = gr.Textbox(
                        label="Ask a Question About the Audio",
                        placeholder="Ask your question about the audio content"
                    )
                    speech_btn = gr.Button("Ask Question", variant="primary")
                    speech_answer = gr.Textbox(label="Answer", lines=8)

            transcript_btn.click(
                processor.get_audio_transcript,
                inputs=audio_input,
                outputs=transcript_output
            )

            speech_btn.click(
                processor.answer_speech_question,
                inputs=[audio_input, speech_question],
                outputs=speech_answer
            )

        with gr.TabItem("Web Search"):
            search_query = gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query"
            )

            with gr.Row():
                with gr.Column():
                    show_links_btn = gr.Button("Start Scrapping", variant="secondary")
                    links_output = gr.Textbox(label="Scraped Links Content", lines=8)

                with gr.Column():
                    search_question = gr.Textbox(
                        label="Ask a Question About Search Results",
                        placeholder="Ask your question about the search results"
                    )
                    search_btn = gr.Button("Ask Question", variant="primary")
                    search_answer = gr.Textbox(label="Answer", lines=8)

            def get_search_content(query):
                success, message = processor.process_search(query)
                if success:
                    return processor.search_context
                return message

            show_links_btn.click(
                get_search_content,
                inputs=search_query,
                outputs=links_output
            )

            search_btn.click(
                processor.answer_search_question,
                inputs=[search_query, search_question],
                outputs=search_answer
            )

if __name__ == "__main__":
    interface.launch(server_name='0.0.0.0', server_port=7860)
