# Data Science Personal Portfolio

Welcome to my Data Science Portfolio! This repository showcases a collection of data science projects I've worked on, highlighting my skills and experience in data analysis, machine learning, and data visualization. Below, you'll find an overview of the projects included in this portfolio, along with instructions on how to navigate and explore them.

## Table of Contents

1. [Project 1: SpaceX Rocket Landing Classification](#project-1-SpaceX-Rocket-Landing-Classification)
2. [Project 2: Mama Cancer Analysis and prediction](#project-2-Mama-Cancer-Analysis-and-prediction)
3. [Project 3: Sunflower Classification with Convolutional Neural Networks](#project-3-Leaf-Diseases-Classification-with-Convolutional-Neural-Networks)
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
11. [Project 11: 10K Project](#project-11-10k-project)
12. [Project 12: GPT Turbo Fine Tunnig](#project-12-gpt-turbo-fine-tunnig)
13. [Project 13: ChatBots back-end and Deployment](#project-13-chatbots-back-end-and-deployment)
14. [Project 14: Complete ML Project - Student Exam Performance Indicator](#project-14-complete-ml-project---student-exam-performance-indicator)


## Project 1: SpaceX Rocket Landing Classification

**Description:**

This Machine Learning project focuses on predicting whether a SpaceX rocket will successfully land on the ground or not, using data obtained through the SpaceX API. The ability to predict the outcome of a rocket landing is essential for ensuring the safety and success of space missions.

**Project Steps:**

1. **Data Acquisition:** The SpaceX API was utilized to gather historical data on rocket landings. These data include relevant information such as launch date and time, rocket type, landing location, and the outcome (success or failure).

2. **Data Processing:** Data from the API often require cleaning and transformation. Data processing techniques were applied to remove null values, encode categorical variables, and adjust the data structure for use in Machine Learning models.

3. **Data Visualization:** Visualizations were carried out to better understand patterns and relationships in the data. Charts and tables were created to display descriptive statistics and trends over time.

4. **Feature Selection:** The most relevant features for predicting rocket landings were identified. This involved feature importance analysis and selecting a suitable set of features for the classification models.

5. **Modeling:** Four different classification models, such as Logistic Regression, Decision Trees, Random Forest, Support Vector Machine were implemented. Each model was trained and evaluated using cross-validation techniques and performance metrics such as accuracy, recall, and F1-score.

6. **Model Evaluation:** The models were compared in terms of their ability to accurately predict whether a SpaceX rocket would successfully land or not. Hyperparameters were tuned, and the model with the best performance was selected.

Technologies Used: Python, Pandas, Seaborn, Scikit-Learn.

**Link to Project:** [SpaceX Rocket Landing Classification](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Classification/SpaceX_classification_prediction.ipynb)

## Project 2: Mama Cancer Analysis and prediction

This Machine Learning project is focused on classifying and analyzing the propensity of an individual to develop breast cancer based on specific variables and features. The goal is to build a predictive model that can assist in early detection and risk assessment of breast cancer, contributing to improved healthcare outcomes.

**Project Steps:**

Data Acquisition: The project begins with the collection of a comprehensive dataset containing information related to breast cancer cases. This dataset typically includes features like age, family history, genetic markers, and medical history.

Data Preprocessing: Data preprocessing techniques are applied to handle missing values, normalize features, and ensure data quality. Additionally, feature engineering may be employed to create new informative variables if necessary.

Exploratory Data Analysis (EDA): EDA is conducted to gain insights into the relationships between different variables and their impact on the likelihood of developing breast cancer. Visualizations and statistical tests are used to uncover patterns and trends.

Feature Selection: The most relevant features for breast cancer prediction are selected. Feature selection methods like recursive feature elimination or feature importance analysis are employed to identify the key variables.

Model Building: Various classification models are implemented, such as Logistic Regression, Support Vector Machine, Random Forest, K-Nearest Neighbors, and Neural Networks. Each model is trained and fine-tuned using techniques like cross-validation.

Model Evaluation: The models are rigorously evaluated using performance metrics such as accuracy, precision, recall, F1-score, and the ROC curve. The aim is to select the model that offers the highest predictive accuracy.

**Technologies Used:** Python, Pandas, Scikit-Learn, Matplotlib, Seaborn, Jupyter Notebook.

**Link to Project:** [Mama Cancer Analysis and prediction](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Classification/mama_cancer_analysis.ipynb)

## Project 3: Sunflower Classification with Convolutional Neural Networks

This Machine Learning project focuses on the detection and classification of diseases in plant leaves using advanced image analysis techniques and Convolutional Neural Networks (CNNs) implemented in TensorFlow. The goal is to develop a system capable of identifying whether a plant leaf is healthy or shows signs of disease, which can be crucial for crop health and agriculture.

**Project Steps:**

Data Collection: A dataset containing images of plant leaves, both healthy and affected by various diseases, was gathered. Each image was appropriately labeled to indicate its health status.

Image Preprocessing: Images underwent preprocessing techniques to resize, orient, and adjust brightness, helping to ensure that CNN models perform optimally.

CNN Model Building: A Convolutional Neural Network (CNN) architecture was designed and implemented using TensorFlow. CNNs are particularly well-suited for image classification tasks due to their ability to capture complex spatial patterns.

Training and Validation: The CNN model was trained using the prepared dataset. Data was split into training and validation sets to monitor model performance and prevent overfitting.

Model Deployment: I created a Flask backend with a html configuration to visualize and finish the project

**Technologies Used:** Python, TensorFlow, Keras, Convolutional Neural Networks (CNNs).

**Link to Project:** [Sunflower Classification with Convolutional Neural Networks](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Neural_Networks/Image_Analysis/Leafs_Diseases_Classification.ipynb)

## Project 4: Concept Analysis of Neural Networks

This project is a comprehensive exploration of various configurations and hyperparameters for Artificial Neural Networks (ANNs) on the Fashion MNIST dataset. The primary objective is to thoroughly investigate how different settings impact the performance of traditional neural networks on this well-known benchmark dataset for image classification.

**Project Steps:**

Dataset Preparation: The Fashion MNIST dataset, comprising 60,000 training images and 10,000 testing images across ten different clothing categories, serves as the foundational dataset for this project.

ANN Architecture Variations: A diverse range of ANN architectures is developed, including networks with different numbers of hidden layers, various activation functions, dropout layers, and different neuron counts. Each architecture is meticulously documented.

Hyperparameter Tuning: A wide range of hyperparameters is explored, encompassing learning rates, batch sizes, optimization algorithms (e.g., SGD, Adam), weight initialization methods, dropout rates, and batch normalization. A systematic approach, such as grid search or random search, is used to evaluate these hyperparameters.

Data Augmentation: Data augmentation techniques, such as rotation, flipping, and scaling, are applied to augment the training dataset. The project assesses the impact of data augmentation on model performance.

Cross-Validation: To ensure robustness, k-fold cross-validation is employed to evaluate model performance across different data splits. This helps identify models with strong generalization capabilities.

Performance Metrics: A range of performance metrics, including accuracy, precision, recall, F1-score, and confusion matrices, are computed and compared for each model configuration and hyperparameter setting.

Visualization: Visualizations are created to illustrate how different ANN architectures and hyperparameters affect the model's ability to recognize clothing items in the Fashion MNIST dataset

**Technologies Used:** Python, Tensorflow, Matplotlib

**Link to Project:** [Concept Analysis of Neural Networks](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Neural_Networks/Image_Analysis/Redes_Neuronales_Analisis.ipynb)

## Project 5: Image Detection Using pretrained YOLOv5

This project involves the fine-tuning of a pretrained YOLOv5 (You Only Look Once version 5) model to detect wind turbines in images. YOLOv5 is a highly efficient and accurate object detection framework, and by fine-tuning it on wind turbine images, we aim to create a specialized model for this specific task.

**Project Steps:**

Data Collection: A dataset of images containing wind turbines is assembled. These images are annotated to indicate the location of wind turbines within each image.

Pretrained Model Selection: A pretrained YOLOv5 model is chosen as the base model. YOLOv5 is known for its speed and accuracy in object detection tasks.

Fine-Tuning: The selected pretrained model is fine-tuned on the wind turbine dataset. During fine-tuning, the model learns to identify wind turbines specifically, adjusting its weights and biases accordingly.

Training: The model is trained on the wind turbine dataset with annotated bounding boxes. Training involves optimizing the model's parameters using techniques like stochastic gradient descent (SGD) or Adam.

Hyperparameter Tuning: Hyperparameters such as learning rate, batch size, and training epochs are tuned to achieve optimal detection performance. This step often involves experimentation and validation to find the best combination.

Evaluation: The fine-tuned model is evaluated using metrics such as mean average precision (mAP), precision, recall, and F1-score. This assessment helps determine the model's accuracy in detecting wind turbines.

Inference: Once trained and evaluated, the model is ready for inference. It can be used to detect wind turbines in new, unlabeled images.

Visualization: Detected wind turbines are visually highlighted in the images, allowing for a qualitative assessment of the model's performance.

**Technologies Used:** Python, Jupyter Notebook, Pandas, Google Colab, Pretrained Models

**Link to Project:** [Image Detection Using pretrained YOLOv5](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Neural_Networks/Image_Analysis/YOLO_Transfer_Learning.ipynb)

## Project 6: Car Price Prediction

This project aims to analyze a dataset containing information about cars and their prices, and then use various regression models from Scikit-Learn to predict the prices of cars based on their features. The objective is to build and evaluate regression models that can provide accurate price estimates for different types of cars.

**Project Steps:**

Data Exploration: The project begins with an in-depth exploration of the dataset. This involves examining the features (independent variables) available in the data, understanding their significance, and checking for missing or anomalous values.

Data Preprocessing: Data preprocessing steps are applied, including handling missing data, encoding categorical variables, and scaling numerical features. This ensures that the data is in a suitable format for machine learning.

Data Visualization: Visualizations, such as scatter plots and histograms, are created to gain insights into the relationships between different features and the target variable (car prices). This helps in understanding the data distribution and identifying potential correlations.

Feature Selection: Feature selection techniques may be applied to identify the most relevant features for price prediction. This step can improve model simplicity and performance.

Model Selection: Various regression models from Scikit-Learn are considered, including Linear Regression, Decision Tree Regression, Random Forest Regression, and Support Vector Regression. Each model is implemented and trained using the preprocessed data.

Hyperparameter Tuning: Hyperparameters for each model are fine-tuned to optimize model performance. Grid search or random search may be employed to systematically explore hyperparameter combinations.

Model Evaluation: The performance of each regression model is evaluated using appropriate metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2). Cross-validation techniques are applied to assess the models' generalization abilities.

Model Comparison: The results of different regression models are compared to identify the model that provides the most accurate price predictions for cars.

At the same time, I created a small backend endpoint with Flask to deploy the model in a local environment

**Technologies Used:** Python, Pandas, Scikit-Learn, Seaborn

**Link to Project:** [Car Price Prediction](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Regression/Car_Price_Analysis_prediction.ipynb)

## Project 7: Real State Price Prediction

This project is designed to analyze a dataset containing information about Indian houses and their prices, and then utilize various regression models from Scikit-Learn to predict the prices of houses based on their features. The project places a strong emphasis on data wrangling and cleaning to ensure the dataset is suitable for accurate price prediction.

**Project Steps:**

Data Collection: Gather a comprehensive dataset that includes details about Indian houses, including attributes like square footage, number of bedrooms, location, amenities, and, most importantly, the selling price.

Data Cleaning: Thoroughly clean and preprocess the data. This step includes handling missing values, removing duplicates, and addressing outliers. Careful consideration is given to ensuring the dataset's quality and consistency.

Data Exploration: Perform exploratory data analysis (EDA) to gain insights into the dataset. Visualizations, such as histograms, scatter plots, and correlation matrices, are used to understand the relationships between features and the target variable (house prices).

Feature Engineering: Create new features or transform existing ones to enhance the dataset's predictive power. This step may involve techniques like one-hot encoding for categorical variables and feature scaling.

Feature Selection: Identify the most relevant features for predicting house prices. Feature selection techniques, such as recursive feature elimination, are applied to optimize the model's performance.

Model Selection: Consider a variety of regression models available in Scikit-Learn, including Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regression, Random Forest Regression, and Gradient Boosting Regression.

Hyperparameter Tuning: Fine-tune the hyperparameters for each regression model to maximize predictive accuracy. Grid search or random search can be employed to explore hyperparameter combinations.

Model Training: Train the selected regression models on the preprocessed dataset using the chosen features.

Model Evaluation: Assess the performance of each regression model using appropriate metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R2). Cross-validation techniques are used to ensure reliable performance evaluation.

Model Comparison: Compare the results of different regression models to identify the model that provides the most accurate predictions of Indian house prices.

**Technologies Used:** Python, Pandas, Scikit-Learn, Seaborn

**Link to Project:** [Real State Price Prediction](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Regression/real_state_price_prediction.ipynb)

## Project 8: Comparison of 3 Deep Learning techniques for Colombian Energy Price Forescasting

This project represents the culmination of your Master's in Artificial Intelligence program, focusing on energy price forecasting in Colombia. The project's main objective is to compare the performance of three different deep learning models—Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), and Transformer—for predicting energy prices.

**Project Steps:**

Data Collection: Gather historical data on energy prices in Colombia. This dataset include previous price trends.

Exploratory Data Analysis (EDA): Perform EDA to gain insights into the data's characteristics. Visualizations and statistical analysis can help identify patterns and correlations in the dataset.

Feature Engineering: Engineer additional features that may enhance the models' predictive capabilities, such as rolling statistics, seasonality, and weather-related features.

Model Selection: Select three deep learning models—LSTM, CNN, and Transformer—for energy price forecasting. Each model is chosen based on its suitability for sequential data and time series forecasting.

Data Splitting: Split the dataset into training, validation, and test sets. Ensure that the temporal order of the data is maintained to simulate real-world forecasting.

Model Development: Develop and train each deep learning model separately using the training data. Tune model-specific hyperparameters and architecture choices for optimal performance.

Model Evaluation: Evaluate the models using appropriate time series forecasting metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and root Mean Squared Error (RMSE). Compare how well each model performs on the validation set.

Ensemble Modeling: Explore the possibility of creating an ensemble model that combines the predictions from all three deep learning models. This can potentially lead to improved forecasting accuracy.

Model Interpretability: Examine the trained models to gain insights into the factors influencing energy price forecasts. Visualize model attention mechanisms in the Transformer model to understand its decision-making process.

Performance Comparison: Summarize and compare the performance of LSTM, CNN, and Transformer models on the test dataset. Assess their strengths and weaknesses in energy price forecasting.

Conclusion and Recommendations: Provide conclusions regarding which model performs best for energy price forecasting in Colombia. Offer recommendations for further improvements or research directions in this domain.

**Technologies Used:** Python, Pandas, Tensorflow, Time Series Analysis.

**Link to Project:** [Comparison of 3 Deep Learning techniques for Colombian Energy Price Forescasting](https://github.com/Jsanchez759/Maestria-Unir/tree/main/TFM)

## Project 9: Data analysis projects

**Description:** These project involves use different dataset with their special conditions and data to do an extensive EDA (Exploratory data analysis) and improve my programming and data analytics skills

**Technologies Used:** Python, Pandas, Matplotlib, Scikit-Learn, Seaborn, Spacial Analysis

**Link to Project:** [Data analysis projects](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Data_Analysis)

## Project 10: NLP models for Sentiment Analysis

**Description:**

This project focuses on sentiment analysis of Amazon Food Reviews, employing two distinct approaches. Initially, the project uses the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis model with the Natural Language Toolkit (NLTK) library to analyze and classify the sentiment of the reviews. Subsequently, a more advanced approach utilizes the pretrained RoBERTa model from the Transformers package to perform sentiment analysis. The aim is to compare the performance of these two methods in gauging the sentiment expressed in the food reviews.

**Project Steps:**

Data Collection: Collect a dataset of Amazon Food Reviews, including text reviews and corresponding sentiment labels (e.g., positive, negative, or neutral).

Data Preprocessing: Clean and preprocess the text data. This includes removing special characters, punctuation, and stop words. Additionally, tokenization is performed to prepare the text for analysis.

VADER Sentiment Analysis: Implement sentiment analysis using the VADER model from the NLTK library. VADER is a lexicon and rule-based sentiment analysis tool that quantifies the sentiment of text into positive, negative, or neutral scores.

Evaluation - VADER: Evaluate the VADER model's performance by comparing its sentiment predictions with the ground truth labels from the dataset. Metrics such as accuracy, precision, recall, and F1-score are computed.

Data Preprocessing for RoBERTa: Prepare the text data in a format suitable for input to the RoBERTa model. This includes tokenization and padding.

RoBERTa Sentiment Analysis: Fine-tune a pretrained RoBERTa model using the transformed dataset for sentiment analysis. RoBERTa is a transformer-based model known for its state-of-the-art performance in various natural language processing tasks, including sentiment analysis.

Evaluation - RoBERTa: Evaluate the RoBERTa model's performance on the same dataset. Compare its sentiment predictions with the ground truth labels and compute evaluation metrics.

Comparison: Analyze and compare the performance of the VADER model and the RoBERTa model. Assess which model provides more accurate sentiment analysis results for Amazon Food Reviews.

Visualization: Create visualizations to illustrate the differences and similarities between the two models' sentiment predictions. This may include word clouds, sentiment distribution plots, and confusion matrices.

**Technologies Used:** Python, Pandas, Matplotlib, Transformers, Seaborn, Natural Language Processing

**Link to Project:** [NLP for Sentiment Analysis](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Neural_Networks/Language_Analysis/Sentiment_Analysis.ipynb)


## Project 11: 10K Project

**Description:**

This project was focus in create a backend code or app that get a pdf file, read it and analyzed it and we can chat with the app to answers questions about this specific pdf file

**Technologies Used:** Python, LangChain, Transformers, Gradio, Streamlit, OpenAI services

**Link to Project:** 
- [Gradio App](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Generative/10K_project/gradio_app.py)
- [Streamlit App](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Generative/10K_project/app/streamlit_app.py)

## Project 12: GPT Turbo Fine Tunnig

**Description:**

The idea with this project was fine-tunning the GPT-3.5 LLM model using the OpenAI services with different datasets and create the pipeline to automate this process and test it 

**Technologies Used:** Python, Transformers, OpenAI services

**Link to Project:** 
- [English Teacher Model](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/Generative/GPT_Turbo_Fine_Tunning/English_Teacher_Model)
- [Jobs Drive Chatbot Model](https://github.com/Jsanchez759/Data-Related-Projects/tree/main/Machine_Learning/Generative/GPT_Turbo_Fine_Tunning/The_Jobs_Driver_Model)

## Project 13: ChatBots back-end and Deployment

**Description:**

The idea with this project was create different chatbots using Gradio as backend and pre-trained LLMs inside serveless end-points, in this project we use the RunPod API to create the instances and the application take this end-point and the user is able to interact and chat with the model

**Technologies Used:** Python, LangChain, Transformers, Gradio, RunPod Services

**Link to Project:** 
- [End-Point Creation](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Generative/text_generation_runpod/runpod_endpoint.ipynb)
- [Chat-Bot Backend](https://github.com/Jsanchez759/Data-Related-Projects/blob/main/Machine_Learning/Generative/text_generation_runpod/gradio_chatbot.py)

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
