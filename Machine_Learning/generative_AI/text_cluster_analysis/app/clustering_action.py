import streamlit as st
import sys
from src.clustering import Clustering
from src.utils import CustomException, logging
import configparser


def clustering_app():
    """
    Main function that sets up the Streamlit application and handles user interactions.
    """
    config = configparser.ConfigParser()
    config.read('src/config/env_config.ini')
    TOKEN = config.get("development", "HUGGINGFACE_KEY")
    MODEL = config.get("development", "MODEL")
    VECTORIZER_MODEL = config.get("development", "VECTOR_MODEL")
    clustering = Clustering(TOKEN, MODEL, VECTORIZER_MODEL)
    # Streamlit interfaces
    st.title("Topic Clustering Application")
    st.markdown(
        f"This app allows you summaries topics using [{MODEL}](https://huggingface.co/{MODEL}) LLM \nfrom HuggingFace space and clustering their vectors using DBSCAN model"
    )
    st.markdown(
        """ 
                Instructions:
                - Choice the number of topics that you want to analyze
                - Write the topic in the corresponding input field
                - Press the button to generate the summaries and clusters
                - Visualize the summaries and clusters in the plot below
                - Visualize the metrics to evaluate the performance of the cluster and summaries
                - Each time you reset the number of topics, the application will reset
                """
    )

    # Streamlit UI components
    try:

        # Store the number of topics in session state
        if "num_topics" not in st.session_state:
            st.session_state.num_topics = 1  # Default value

        # Use the stored number of topics when displaying the input
        num_topics = st.number_input(
            "Enter the number of topics",
            min_value=1,
            value=st.session_state.num_topics,
            step=1,
        )

        # Save the number of topics and initialize the list of topics if it doesn't exist or if different with the previous state
        if num_topics != st.session_state.num_topics:
            st.session_state.num_topics = num_topics
            st.session_state.topics = [
                "" for _ in range(num_topics)
            ]  # Reset topics if the number changes

        if "topics" in st.session_state:
            for i in range(st.session_state.num_topics):
                st.session_state.topics[i] = st.text_input(
                    f"Topic {i+1}", st.session_state.topics[i]
                )

        if st.button("Save values and generate the clusters of the topics"):
            if len(st.session_state.topics) > 1:
                with st.spinner("Generating summaries and clusters..."):
                    # Generate summaries
                    summaries = clustering.generate_summaries(st.session_state.topics)
                    st.markdown("---")
                    vectors_scaled, cluster_labels, n_clusters_ = (
                        clustering.perform_clustering(summaries)
                    )
                    st.markdown("**Evaluation the results**")
                    st.markdown("---")
                    clustering.evaluate_clusters_results(vectors_scaled, cluster_labels, n_clusters_)

            else:
                st.warning(
                    "Please write more than 1 topic to generate summaries and clusters",
                    icon="⚠️",
                )

    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error: {e}")
        raise CustomException(e, sys)
