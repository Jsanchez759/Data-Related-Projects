from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_qa_chain(llm):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """

    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    Question: {question}
    """
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    chain = prompt_template | llm | StrOutputParser()

    return chain


def create_summary_chain(llm):
    """
    Create an LLMChain for generating summaries.

    :param llm: Language model instance
    :return: LLMChain instance
    """
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.
    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.
    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:
    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt_template = PromptTemplate(input_variables=["transcript"], template=template)
    chain = prompt_template | llm | StrOutputParser()

    return chain
