## Fall 2018 DATA622.001 Homework #2
- Assigned on September 4, 2018
- Due on September 25, 2018 12:00 PM EST
- 15 points possible, worth 15% of your final grade

### Required Reading
- Read the [Introduction Chapter](https://www.deeplearningbook.org/contents/intro.html) of the Deep Learning Book
- Read [Chapter 5](https://www.deeplearningbook.org/contents/ml.html) of the Deep Learning Book

### Data Pipeline using Python (13 points total)

Build a data pipeline in Python that downloads data using the urls given below, trains a classification model of your choice on the training dataset using sklearn and scores the model on the test dataset.

#### Scoring Rubric
The homework will be scored based on code efficiency (hint: use functions, not stream of consciousness coding), code cleanliness, code reproducibility, and critical thinking (hint: commenting lets me know what you are thinking!)  

#### Instructions:
tl;dr: Submit the following 5 items on hw2 github repo.  
- answer the questions under the "Critical Thinking" section below
- upload a "requirements.txt" file to repo
- upload a "pull_data.py" file to repo
- upload a "train_model.py" file to repo
- upload a "score_model.py" file to repo

More details:

- <b> requirements.txt </b> 

This file documents all python package dependencies needed to run the rest of your python scripts.  You can generate this file by using pipenv/virtual env, following the tutorial [here](https://docs.python-guide.org/dev/virtualenvs/) and [here](https://docs.python-guide.org/dev/virtualenvs/#installing-packages-for-your-project).  Or, if you are a Anaconda user, the Anaconda equivalent can be found [here](https://conda.io/docs/commands.html#conda-vs-pip-vs-virtualenv-commands).  The idea of this is that, when you call upon `pip install -r requirements.txt` or `conda install --yes --file requirements.txt`, this will install all python packages needed to run the .py files below.  (hint: `pip freeze`)

- <b> pull_data.py </b> 

When this script is called using `python pull_data.py` in the command line, this will load the 2 dataset files below either by using the url as input and pull directly from online OR by loading the locally saved .csv files.  If using the former method, take care to not upload Kaggle authentication details (aka secrets) in your homework submission. There <b>must</b> be a data check step to ensure the data has been pulled correctly and clear commenting and documentation for each step inside the .py file.

    Training dataset url: https://www.kaggle.com/c/titanic/download/train.csv
    Scoring dataset url: https://www.kaggle.com/c/titanic/download/test.csv

- <b> train_model.py </b> 

When this is called using `python train_model.py` in the command line, this will take in the training dataset csv, perform the necessary data cleaning and imputation, and fit a classification model of your choice to the dependent Y.  There must be data check steps and clear commenting for each step inside the .py file.  The output for running this file is the classification model saved as a .pkl file in the local directory.  

- <b> eda.ipynb </b> 

This supplements the commenting inside train_model.py.  This is the place to provide scratch work and plots to convince me why you did certain data imputations and manipulations inside the train_model.py file.  Remember that the thought process and decision for why you chose the final model must be clearly documented in this section.  

- <b> score_model.py </b> 

When this is called using `python score_model.py` in the command line, this will ingest the .pkl random forest file and apply the model to the locally saved scoring dataset csv.  There must be data check steps and clear commenting for each step inside the .py file.  The output for running this file is a csv file with the predicted score, as well as a png or text file output that contains the model accuracy report (e.g. sklearn's classification report or any other way of model evaluation).  


### Critical Thinking (2 points total)

Modify this ReadMe file to answer the following questions directly in place.

1. Why did we have to write in Python scripts instead of keeping everything in a Jupyter Notebook?  Read [this](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/) for some inspiration. (0.5 points)

Jupyter notebooks and RMarkdown files have their time and place in data science. They are great for exploratory data analysis and making presentations. However, the more frequently you need to update your analysis with new data, the less convenient they are. They are completely unsuited for the daily training and scoring of a machine learning model. Contructing an automated data pipeline is a more appropriate way to accomplish this task. Speaking from experience, creating a data pipeline to run without failing is a skill unto itself and cannot be learned by using Notebooks.

2. What are some things that could go wrong with our current pipeline that we are not able to address?  For your reference, read [this](https://snowplowanalytics.com/blog/2016/01/07/we-need-to-talk-about-bad-data-architecting-data-pipelines-for-data-quality/) for inspiration. (0.5 points)

Since I hosted the code on Github, events that could interupt our data pipelinie include the following:

- The token needed to access the data via the Classroom product could expire.

- Github could alter their domain's URL.

- If the repository had a collaborator, he or she could change the code unexpectedly. When we tried to run the code, it may not do what we originally intended.

3. How would you build things differently if this dataset was 1 trillion+ rows? (0.5 points)

In this scenario, I would build the pipeline to read and write the data with a database. I would not store any of the data locally. I would have to spend time microbenchmarking different pieces of code so that the model would not take longer than needed. Additionally, I prefer not to run the model on my local machine, but on a more powerful remote server. If parallel processing was an option, that would also be helpful in this situation. 

4. How would you build things differently if the testing/scoring dataset on Kaggle was actually changing constantly and need to be frequently re-pulled/downloaded? (0.5 points)

The biggest difference is that I would want the script to be scheduled to run periodically, so that it would not require manual triggering. This could be accomplished by creating a cron job or using Windows Task Scheduler, ideally on a remote machine or server so that the local machine would not have to be running constantly.