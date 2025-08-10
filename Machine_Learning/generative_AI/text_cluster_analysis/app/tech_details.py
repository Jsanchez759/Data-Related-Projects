import streamlit as st


def tech_implementation():
    st.title("Technical Implementations")
    st.markdown(
        """
    ### Clustering and summarization Implementation
    - **Model**: Uses `Qwen2-1.5B-Instruct` from Hugging Face to generate topic summaries.
        - We choose this model because is a very good option in terms of size vs. performance, considering this application is for now in a local environment
          and not a production environment with low resources. With just 1.5B of parameters this model has demonstrated competitiveness against proprietary models across a series of benchmarks compared with models inside the same range of parameters
          according to their [Blog](https://qwenlm.github.io/blog/qwen2/).
          Also, is available under the Apache 2.0 License, allowing for free and open-source use for research and commercial applications.
          [HuggingFace card](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct).

    - **Vectorizer**: Uses `bert-base-uncased` to vectorize the summaries.
        - We choose a BERT model, more specific bert-base-uncased [HuggingFace card](https://huggingface.co/google-bert/bert-base-uncased) for vectorization,
          this model is widely used for natural language processing tasks and has achieved good results in various tasks.

    - **Clustering Algorithm**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) for clustering the vectorized summaries.
        - We choose DBSCAN to analyze the vectors and perform clustering, this model can identify clusters of arbitrary shapes,
          unlike K-means, which assumes spherical clusters. This makes DBSCAN suitable for text data where clusters might not be uniformly distributed or might have irregular boundaries.
          For the most important hyperparameters for this DBSCAN model:
            - eps: After standarized the vectors, we calculate the average of the distances between each vector to improve how me manage the hyperparameter
            - min_samples: We put this as a fix value of 2, which means that a point needs to have at least 2 neighbors to be considered a core point.
            - Finally, we use a PCA dimensionality reduction technique to visualize the clusters and summarize the vectors.

    ---
    ### Evaluation Metrics

    - **Clustering evaluation**:
        - **Silhouette metric**: Refers to a method of interpretation and validation of consistency within clusters of data. The technique provides a succinct graphical representation of how well each object has been classified.
          Calculating the average silhouette score of all points, we get a measure of the overall cohesion and separation of the clustering.
          and helps us to understand whether the formed clusters are truly meaningful or if the data lacks a clear cluster structure.
          The silhouette score for each data point is calculated as follows:
            -  a: The mean distance between the point and all other points in the same cluster.
            -  b: The mean distance between the point and all other points in the nearest neighboring cluster.
            -  s = (b - a) / max(a, b)
          ###### Range of Values and Interpretation
            - 0.71 - 1.00: Very Good: Points are well-clustered, and the clusters are clearly distinct.
            - 0.51 - 0.70: Good: Points are reasonably well-clustered, though some overlap between clusters may exist.
            - 0.26 - 0.50: Acceptable: Clusters are formed but with considerable overlap; some clusters may not be meaningful.
            - ≤ 0.25: Poor: Clusters are indistinguishable; the data may not have a clear clustering structure, or the chosen number of clusters is inappropriate.

            [Silhouette documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)


    - **Summary evaluation**:
        - **ROUGE (Recall-Oriented Understudy for Gist) metric**: Evaluates the quality of the generated summaries by comparing them with a set of reference summaries.
          The ROUGE metric is a set of metrics designed to evaluate the quality of text summaries.
          We use the ROUGE-1 (This measures the overlap of each word between the system and reference summaries.), and ROUGE-L (which measures the longest common subsequence) metrics to assess the performance of the generated summaries.

          - ROUGE-1:
            - High ROUGE-1 score (> 0.8): Indicates a very good overlap between the system and reference summaries, suggesting that the generated summaries are highly comprehensive and accurate.
            - Moderate ROUGE-1 score (0.6 - 0.8): Indicates a good overlap between the system and reference summaries, but there may be some areas for improvement.
            - Low ROUGE-1 score (< 0.6): Suggests that the generated summaries are not very comprehensive or accurate, and there is a significant area for improvement.

          - ROUGE-L:
            - High ROUGE-L score (> 0.8): Indicates a very good overlap between the system and reference summaries, suggesting that the generated summaries are highly comprehensive and accurate.
            - Moderate ROUGE-L score (0.6 - 0.8): Indicates a good overlap between the system and reference summaries, but there may be some areas for improvement.
            - Low ROUGE-L score (< 0.6): Suggests that the generated summaries are not very comprehensive or accurate, and there is a significant area for improvement.
          
          [ROUGE documentation](https://pypi.org/project/rouge-score/)


        - **BERT metrics**: Evaluates summaries by measuring the semantic similarity between generated and reference summaries using pre-trained BERT embeddings.
        Each token in the reference sentence is matched to the most similar token in the generated sentence, and vice versa. This process is used to calculate the precision, recall, and F1 score, which are measures of the quality of the generated text.

           - Precision:
             - High precision (> 0.85): The generated text is very similar to the reference text, but it might miss some important information.
             - Moderate precision (0.60 - 0.85): The generated text has a decent overlap with the reference text but may lack detail.
             - Low precision (< 0.60): The generated text is quite different from the reference text and may not capture the key points well.

          - Recall:
            - High recall (> 0.85): The generated text covers most of the information in the reference text, possibly at the cost of verbosity.
            - Moderate recall (0.60 - 0.85): The generated text covers a reasonable amount of information from the reference text.
            - Low recall (< 0.60): The generated text misses significant portions of the reference text.

          - F1-Score:
            - High F1-Score (> 0.85): Indicates a good balance between precision and recall, meaning the generated text is both accurate and comprehensive.
            - Moderate F1-Score (0.60 - 0.85): Indicates that the generated text is somewhat accurate and comprehensive but could be improved.
            - Low F1-Score (< 0.60): Suggests that the generated text may either miss important details or include irrelevant information.

          [BERT documentation](https://pypi.org/project/bert-score/)

    ---

    ### Error Handling and Logging
    - **Error handling**: We use a try-except block to catch any exceptions that may occur during the execution of the application.
      If an error occurs, it will be logged and displayed to the user, allowing for easier debugging and troubleshooting.

    - **Logging**: We use the `logging` module from the Python standard library to log events and errors during the execution of the application.
      The logs will be stored in the `logs` directory, with each log file named after the current date and time.
      This allows for easy monitoring and analysis of the application's behavior over time.

    ---

    ### Code Structure
    - `app.py`: Main Streamlit application code, handles the user interface and integration with the `Clustering` class.
    - `tech_details.py`: Code to explain all the technical details for this app.
    - `clustering_action.py`: Code to initialize the `Clustering` class and perform all the steps
    - `src`: Contains auxiliary code and modules, including:
        - `clustering.py`: Contains the `Clustering` class which handles summary generation, clustering and evaluation.
        - `utils.py`: Utility functions used in the application regarding logs and error handling
    - `requirements.txt`: Lists all dependencies required for the application.

    ---

    ### Directory Structure
    - `app`: Contains the main application code.
    - `src`: Contains auxiliary code and modules.
    - `logs`: Directory for log files.
    """
    )
