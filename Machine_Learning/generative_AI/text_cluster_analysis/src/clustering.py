import streamlit as st
import plotly.express as px
import numpy as np
import torch
import pandas as pd
import time
import sys
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.utils import CustomException, logging
import wikipediaapi
from rouge_score import rouge_scorer
from bert_score import BERTScorer

class Clustering:
    def __init__(self, token: str, model: str, vectorizer_model: str):
        """
        Initialize Clustering object with models parameters to summarize data and get the clusters.

        Args:
        - token (str): Token for the text generation pipeline.
        - model (str): Pre-trained model for text generation.
        - vectorizer_model (str): Pre-trained model for feature extraction.

        Returns:
        - None
        """
        self.TOKEN = token
        self.MODEL = model
        self.VECTORIZER_MODEL = vectorizer_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Function to generate summaries using a pre-trained model
    def generate_summaries(self, topics: list) -> list:
        """
        Generate detailed summaries for given topics using a pre-trained model.

        Args:
        - topics (list): List of topics (strings) to generate summaries for.

        Returns:
        - summaries (list): List of generated summaries (strings) for each topic.
        """
        start_time = time.time()
        # Get and download the model from HuggingFace
        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL, token=self.TOKEN, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL,
            torch_dtype=torch.bfloat16,
            token=self.TOKEN,
        )
        # Get and use the template for the model
        try: 
            topics_templates = [
                tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful and factual assistant.",
                        },
                        {
                            "role": "user",
                            "content": f"You are a precise and factual assistant. Please provide a brief, accurate summary of the following topic, focusing only on well-known facts and avoiding speculation or invented information: \nTopic: {topic} \nRemember, the goal is to avoid any hallucination or fabrication. If you are unsure about something, it's better to omit it.",
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for topic in topics
            ]

            model_inputs = tokenizer(
                topics_templates, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,  # Lower temperature for focused output
                top_k=50,  # Top-k sampling for safer choices
                top_p=0.9,  # Top-p sampling to limit risky outputs
                repetition_penalty=1.2,  # Discourage repetition
                num_return_sequences=1,  # Return one sequence
            )
            generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

            summaries = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in generated_ids
            ]

            for index, text in enumerate(summaries):
                st.markdown(f"**Summary {index+1}:** {text}")

            print(time.time() - start_time, "seconds generating summaries")
        except Exception as e:
            st.error(f"Error getting the summaries of the topics: {e}")
            logging.error(f"Error getting the summaries of the topics: {e}")
            raise CustomException(e, sys)

        return summaries

    # Function for DBSCAN clustering
    def perform_clustering(self, summaries: list) -> None:
        """
        Perform DBSCAN clustering on the summaries' feature vectors and visualize clusters.
    
        Args:
        - summaries (list): List of generated summaries (strings) to cluster.

        Returns:
        - cluster_labels (list): List of cluster labels assigned to each summary.
        """
        # Feature pipeline using a specific pre-trained language model for text vectorization
        start_time = time.time()
        vectorizer = pipeline(
            "feature-extraction",
            model=self.VECTORIZER_MODEL,
            tokenizer=self.VECTORIZER_MODEL,
        )
        # Computes the mean of the extracted feature vectors and flattens it into a single vector
        try:
            vectors = np.array([np.mean(vectorizer(text), axis=1).flatten() for text in summaries])
            topics = [f"Topic {i+1}" for i in range(len(summaries))]

            # The feature vectors are scaled, this help make the distances more manageable and the algorithm work more effectively.
            scaler = StandardScaler()
            vectors_scaled = scaler.fit_transform(vectors)

            # We calculate the mean of distances between the scaled feature vectors and this will be the eps to better adapt to the shape of the vectors
            distances = pairwise_distances(vectors_scaled)
            dbscan = DBSCAN(eps=distances.mean() + 5, min_samples=2)
            cluster_labels = dbscan.fit_predict(vectors_scaled)
            labels = dbscan.labels_
            # evaluate the cluster using silhouette score
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Get the number of clusters ignoring the noisy (-1)

            # Principal Component Analysis (PCA) is applied to the scaled feature vectors to reduce their dimensionality to 2 components and improve the visibility
            pca = PCA(n_components=2)
            components = pca.fit_transform(vectors_scaled)
            print(time.time() - start_time, "seconds getting cluster summaries")

            # We create a pandas dataframe to visualize the clusters and their corresponding topics
            df = pd.DataFrame(components, columns=["Dimension 1", "Dimension 2"])
            df["Cluster"] = [f"Cluster {label}" for label in cluster_labels]
            df["Topic"] = topics

            fig = px.scatter(
                df,
                x="Dimension 1",
                y="Dimension 2",
                color="Topic",
                text="Cluster",
                title=f"Cluster Visualization by Topic - Estimated number of cluster {n_clusters_}",
                labels={"Dimension 1": "PCA Dimension 1", "Dimension 2": "PCA Dimension 2"},
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error calculating the cluster: {e}")
            logging.error(f"Error calculating the cluster: {e}")
            raise CustomException(e, sys)

        return vectors_scaled, cluster_labels, n_clusters_

    # Function to evaluate the clustering results using silhouette score
    def evaluate_clusters_results(self, vectors: np.array, cluster_labels: list, n_clusters_: int) -> None:
        """
        Evaluate the generated clusters using Silhouette metric

        Args:
        - vectors (numpy array): array of the vectors generated by the summaries.
        - cluster_labes (list): list of cluster generated.
        - n_clusters_ (int): number of clusters

        Returns:
        - None

        """
        try: 
            st.markdown(f"**Evaluate the clusters:**")
            if n_clusters_ > 1:
                silhouette_avg = silhouette_score(vectors, cluster_labels)
                st.markdown(f"**Silhouette Score of the clusters:** {silhouette_avg:.2%}")
                if silhouette_avg < 0.25:
                    st.text("The clusters may not be well-separated or may contain outliers.")
                elif silhouette_avg >= 0.26 and silhouette_avg < 0.5:
                    st.text("Clusters are formed but with considerable overlap;.")
                elif silhouette_avg >= 0.51 and silhouette_avg < 0.7:
                    st.text("Points are reasonably well-clustered, though some overlap between clusters may exist.")
                else:
                    st.text("Clusters are well-separated and well-clustered.")
            else:
                st.write("Silhouette Score: Cannot be computed because the number of clusters is less than 2.")
        except Exception as e:
            st.error(f"Error calculating the silhouettescore: {e}")
            logging.error(f"Error calculating the silhouettescore: {e}")
            raise CustomException(e, sys)

    def get_wikipedia_summary(self, topics: list) -> list:
        """
        Get wikipedia summaries for given topics.

        Args:
        - topics (list): List of topics (strings) to generate summaries for.

        Returns:
        - first_paragraph (list): The first paragraph of the Wikipedia page for the topic.
        """
        try:
            wiki_wiki = wikipediaapi.Wikipedia('project' ,"en")
            actual_summaries = [wiki_wiki.page(topic).summary[:512] if wiki_wiki.page(topic).exists() else
            f"Wikipedia page not found for topic: {topic}" for topic in topics]
            summaries_url = [wiki_wiki.page(topic).fullurl if wiki_wiki.page(topic).exists() else
            f"Wikipedia page not found for topic: {topic}" for topic in topics]
        except Exception as e:
            st.error(f"Error getting the actual summary: {e}")
            logging.error(f"Error getting the actual summary: {e}")
            raise CustomException(e, sys)

        return actual_summaries, summaries_url

    # Function to evaluate the generated summaries
    def evaluate_summaries_results(self, summaries: list, actual_summaries: list, summaries_url: list) -> None:
        """
        Evaluate the generated summaries by displaying them in a list.

        Args:
        - summaries (list): List of generated summaries (strings).
        - actual_summaries (list): List of actual summaries (strings).
        - summaries_url (list): List of actual summaries url (strings).

        Returns:
        - None
        """
        try:
            st.markdown(f"**Evaluate the generated summaries of the LLM:**")
            rouge_scorer_object = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            bert_scorer = BERTScorer(model_type='bert-base-uncased')
            for gen_summary, ref_summary, index, url in zip(summaries, actual_summaries, range(len(summaries)), summaries_url):
                st.markdown(f"**Actual Summary {index+1}:** {ref_summary}")
                if ref_summary.startswith("Wikipedia page not found"):
                    continue
                st.markdown(f"**Source:** {url}")
                score = rouge_scorer_object.score(ref_summary, gen_summary)
                st.write(f"ROUGE-1 score: {score["rouge1"].fmeasure:.2%}")
                st.write(f"ROUGE-L score: {score["rougeL"].fmeasure:.2%}")

                precision, recall, f1_score = bert_scorer.score([gen_summary], [ref_summary])
                st.write(f"F1-BERT Score: {f1_score.mean():.2%}")
                st.write(f"Precision-BERT Score: {precision.mean():.2%}")
                st.write(f"Recall-BERT Score: {recall.mean():.2%}")
                st.markdown("---")
            st.write("Please refer to technical Implementations page for more information about the evaluation metrics and how to interpret them")
        except Exception as e:
            st.error(f"Error calculating the metrics to evaluate the summaries: {e}")
            logging.error(f"Error calculating the metrics to evaluate the summaries: {e}")
            raise CustomException(e, sys)