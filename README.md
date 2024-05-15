# Data Science Personal Portfolio

Welcome to my Data Science Portfolio! This repository showcases a collection of data science projects I've worked on, highlighting my skills and experience in data analysis, machine learning, and data visualization. Below, you'll find an overview of the projects included in this portfolio, along with instructions on how to navigate and explore them.

## Table of Contents

1. [Project 1: SpaceX Rocket Landing Classification](#project-1-SpaceX-Rocket-Landing-Classification)
2. [Project 2: Mama Cancer Analysis and prediction](#project-2-Mama-Cancer-Analysis-and-prediction)
3. [Project 3: Sunflower Classification with Convolutional Neural Networks](#Project-3-Sunflower-Classification-with-Convolutional-Neural-Networks)
4. [Project 4: Concept Analysis of Neural Networks](#project-4-Concept-Analysis-of-Neural-Networks)
5. [Project 5: Image Detection Using pretrained YOLOv5](#project-5-Image-Detection-Using-pretrained-YOLOv5)
6. [Project 6: Car Price Prediction](#project-6-Car-Price-Prediction)
7. [Project 7: Real State Price Prediction](#project-7-Real-State-Price-Prediction)
8. [Project 8: Comparison of 3 Deep Learning techniques for Colombian Energy Price Forescasting](#project-8-Comparison-of-3-Deep-Learning-techniques-for-Colombian-Energy-Price-Forescasting)
9. [Project 9: Data analysis projects](#project-9-Data-analysis-projects)
    - Energy Consume Analysis
    - Hotel Booking Analysis
    - Uber Data Analysis
10. [Project 10: NLP models for Sentiment Analysis](#project-10-nlp-models-for-sentiment-analysis)
11. [Project 11: Talk with your Project](#Project-11-Talk-with-your-Project)
12. [Project 12: GPT Turbo Fine Tunnig](#project-12-gpt-turbo-fine-tunnig)
13. [Project 13: ChatBots back-end and Deployment](#project-13-chatbots-back-end-and-deployment)
14. [Project 14: Complete ML Project - Student Exam Performance Indicator](#project-14-complete-ml-project---student-exam-performance-indicator)


## Project 1: SpaceX Rocket Landing Classification

**Description:**

This Machine Learning project focuses on predicting whether a SpaceX rocket will successfully land on the ground or not, using data obtained through the SpaceX API. With this project I gained a great knowledge about classification task, their performance metrics and models variations. At the same time I created a complete Flask app to consume the model with different data preparation techniques and a ready Dockerfile to deploy the model.

**Technologies Used:** Python, Pandas, Seaborn, Scikit-Learn, Flask, Docker

**Link to Project:** [SpaceX Rocket Landing Classification](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/Classification/spaceX_classification_project)

## Project 2: Mama Cancer Analysis and prediction

This Machine Learning project is focused on classifying and analyzing the propensity of an individual to develop breast cancer based on specific variables and features. This is a comprehensive project where an complete EDA is conducted to gain insights into the relationships between different variables and their impact on the likelihood of developing breast cancer. Visualizations and statistical tests are used to uncover patterns and trend, also Various classification models are implemented, such as Logistic Regression, Support Vector Machine, Random Forest, K-Nearest Neighbors, and Neural Networks. Each model is trained and fine-tuned using techniques like cross-validation.

**Technologies Used:** Python, Pandas, Scikit-Learn, Matplotlib, Seaborn, Jupyter Notebook.

**Link to Project:** [Mama Cancer Analysis and prediction](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Classification/mama_cancer_analysis.ipynb)

## Project 3: Sunflower Classification with Convolutional Neural Networks

This Machine Learning project focuses on classification of types of Sunflower using advanced image analysis techniques and Convolutional Neural Networks (CNNs) implemented in TensorFlow. This is a complete project that start with data preparation, model testing and developing and finally the developing a Flask app to interact with the model and visualize their prediction.

**Technologies Used:** Python, TensorFlow, Keras, Convolutional Neural Networks (CNNs), Flask

**Link to Project:** [Sunflower Classification with Convolutional Neural Networks](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/Neural_Networks/Image_Analysis/flower_classification)

## Project 4: Concept Analysis of Neural Networks

This project is a comprehensive exploration of various configurations and hyperparameters for Artificial Neural Networks (ANNs) on the Fashion MNIST dataset. The primary objective is to thoroughly investigate how different settings impact the performance of traditional neural networks on this well-known benchmark dataset for image classification. In this project a diverse range of ANN architectures is developed, including networks with different numbers of hidden layers, various activation functions, etc, I tried the performance of wide range of hyperparameters, techniques such data augmentation and cross-validation are also reviewed

**Technologies Used:** Python, Tensorflow, Matplotlib

**Link to Project:** [Concept Analysis of Neural Networks](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Neural_Networks/Image_Analysis/Redes_Neuronales_Analisis.ipynb)

## Project 5: Image Detection Using pretrained YOLOv5

This project involves the fine-tuning of a pretrained YOLOv5 (You Only Look Once version 5) model to detect wind turbines in images. YOLOv5 is a highly efficient and accurate object detection framework, and by fine-tuning it on wind turbine images, we aim to create a specialized model for this specific task. The fine-tuned model is evaluated using metrics such as mean average precision (mAP), precision, recall, and F1-score. This assessment helps determine the model's accuracy in detecting wind turbines.

**Technologies Used:** Python, Jupyter Notebook, Pandas, Google Colab, YOLOv5

**Link to Project:** [Image Detection Using pretrained YOLOv5](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Neural_Networks/Image_Analysis/YOLO_Transfer_Learning.ipynb)

## Project 6: Car Price Prediction

This project aims to analyze a dataset containing information about cars and their prices, and then use various regression models from Scikit-Learn to predict the prices of cars based on their features. The objective is to build and evaluate regression models and gain knowledege about all the necessary steps to develop a complete regression model, since data exploration, preprocessing, cleaning, exploratory data analysis, feature selection, model developing and comparison using appropriate metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2). Cross-validation techniques are applied to assess the models' generalization abilities. At the same time.

**Technologies Used:** Python, Pandas, Scikit-Learn, Seaborn

**Link to Project:** [Car Price Prediction](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Regression/car_price_prediction/Car_Price_Analysis_prediction.ipynb)

## Project 7: Real State Price Prediction

This project is designed to analyze a dataset containing information about Indian houses and their prices, and then utilize various regression models from Scikit-Learn to predict the prices of houses based on their features. The project places a strong emphasis on data wrangling and cleaning to ensure the dataset is suitable for accurate price prediction. At the same time, I created a small backend endpoint with Flask to deploy the model in a local environment

**Technologies Used:** Python, Pandas, Scikit-Learn, Seaborn

**Link to Project:** [Real State Price Prediction](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/Regression/real_state_app)

## Project 8: Comparison of 3 Deep Learning techniques for Colombian Energy Price Forescasting

This project represents the culmination of my Master's in Artificial Intelligence program, focusing on energy price forecasting in Colombia. The project's main objective is to evaluate 3 deep learning architectures (LSTM, CNN and Transformers) in energy cost modeling for three different time periods, to determine which technique is the most appropriate for modeling and price prediction. Also a base model and a strategy for varying different hyperparameters of the chosen models were proposed to finally choose the most appropriate technique according to the results of each time window and each architecture.

**Technologies Used:** Python, Pandas, Tensorflow, Time Series Analysis.

**Link to Project:** [Comparison of 3 Deep Learning techniques for Colombian Energy Price Forescasting](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/master_degree_project/project)

## Project 9: Data analysis projects

**Description:** These project involves use different dataset with their special conditions and data to do an extensive EDA (Exploratory data analysis) and improve my programming and data analytics skills

**Technologies Used:** Python, Pandas, Matplotlib, Scikit-Learn, Seaborn, Spacial Analysis

**Link to Project:** [Data analysis projects](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Data_Analysis)

## Project 10: NLP models for Sentiment Analysis

**Description:**

This project focuses on sentiment analysis of Amazon Food Reviews, employing two distinct approaches. Initially, the project uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis model with the Natural Language Toolkit (NLTK) library to analyze and classify the sentiment of the reviews. Subsequently, a more advanced approach utilizes the pretrained RoBERTa model from the Transformers package to perform sentiment analysis. The aim is to compare the performance of these two methods in gauging the sentiment expressed in the food reviews.

**Technologies Used:** Python, Pandas, Matplotlib, Transformers, Seaborn, Natural Language Processing

**Link to Project:** [NLP for Sentiment Analysis](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Neural_Networks/Language_Analysis/Sentiment_Analysis.ipynb)


## Project 11: Talk with your Project

**Description:**

This project was focus in create an app that get a PDF file, analyzed it and have the ability to interact with us using the info inside the PDF, the backend code to analyze the text was created using the LangChain and Transformers library with open sources models and advance NLP techniques, to create the user interaction, I created 2 apps with Gradio and Streamlit

**Technologies Used:** Python, LangChain, Transformers, Gradio, Streamlit, OpenAI services

**Link to Project:** 
- [Gradio App](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Generative/10K_project/gradio_app.py)
- [Streamlit App](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Generative/10K_project/app/streamlit_app.py)

## Project 12: GPT Turbo Fine Tunnig

**Description:**

The idea with this project was fine-tunning the GPT-3.5 LLM model using the OpenAI services with different datasets and create the pipeline to automate this process, test the new models and estimate the costs of this training. 

**Technologies Used:** Python, Transformers, OpenAI services

**Link to Project:** 
- [English Teacher Model](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/Generative/GPT_Turbo_Fine_Tunning/English_Teacher_Model)
- [Jobs Drive Chatbot Model](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/Generative/GPT_Turbo_Fine_Tunning/The_Jobs_Driver_Model)

## Project 13: ChatBots back-end and Deployment

**Description:**

The idea with this project was create different chatbots using Gradio as backend and pre-trained LLMs inside serveless end-points, in this project we use the RunPod API to create the instances and the application take this end-point and the user is able to interact and chat with the model using a Gradio App

**Technologies Used:** Python, LangChain, Transformers, Gradio, RunPod Services

**Link to Project:** 
- [LLM Chatbot](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/Generative/text_generation_runpod)

## Project 14: Complete ML Project - Student Exam Performance Indicator

**Description:**

Complete Machine Learning project, where the idea was simulate the behaviour and configuration of a complete and industry project,
here we have a replication of folders and using more advanced techniques such as logger, handling execptions, pipelines and use of classes inside the project, also we create a Dockerfile to deploy the final web app

**Technologies Used:** Python, Scikit-Learn, Docker

**Link to Project:** 
- [Project](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/complete_project)

## Contact Information

If you have any questions or would like to discuss collaboration opportunities, please feel free to contact me:

- **Email:** jesussanchezd@hotmail.es
- **LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/in/jesus-sanchez-data-science/)

Thank you for visiting my Data Science Portfolio, and I hope you find the projects informative and insightful!
