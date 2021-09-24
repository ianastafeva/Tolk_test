# Tolk_test

## Description of the developed software

The project is designed to allow users (customers) to train more (fine-tune) pre-trained classification models (classification of chatbot intentions) with a personalized dataset by interacting with an API built to let the users select the pre-trained model (chatbot) based on domain, upload their new dataset and interact with the newly fine-tuned chatbot. The flowchart below illustrates an overview of the project process.   

![flowchart](https://github.com/ianastafeva/Tolk_test/blob/8f6ee04043d5e35d373d4cde858b9a8efde844f3/Chatbot_API_flowchart.png?raw=true)

It should be mentioned that the preprocessing block of the project is only envisioned and not implemented because there is an NLP difficulty present here. The difficulty is how to restructure the new dataset (especially if there are no strict rules on the formating of the dataset) in the same way as the original dataset used to build the model selected by the user. What I mean by that is, for example, if the user dataset is plain text, how to extract the relevant information such as intention, patterns, and responses from such text. One solution is to develop an NLP model that can recognize common similarities in the dataset, split the dataset into classes (intentions) based on the similarity, then restructure each class to patterns and responses. I am not sure if such solution is actually feasible as I didn't have enough time to test its applicability. 

The difficulty is not only specific for the model in this project as it is also present in more advanced ones such as RASA NLU where the set of labels (intents, actions, entities, and slots) for which the base model is trained should be exactly the same as the ones present in the training data used for fine-tuning.

Finally, when testing the fine-tuned model (chatbot) through the API, one can notice that the responses are stange and that because our integration of the chatbot with the API is not accurate.  


## Set up
Clone or download the project files to local device and unzip it

### The user has ananconda python in his/her device
1) Open termainal (Mac, Linux) or anaconda prompt as administrator (windows)

2) Move to the directory containing the unziped project files using 'cd' command

3) Create an new virtual environment with the following command:

   >conda create -n chatbot_api python=3.8.5

4) Activate the environment with the following command:

   >conda activate chatbot_api

5) Install the needed python packages with the following command:

   >pip install -r requirements.txt

6) Download needed nltk files by the following commands:
   >python
   
   >import nltk 
   
   >nltk.download()
 
 In the GUI window that opens simply press the 'Download' button to download all corpora

Now the user is set to run the codes of the projects

### The user doesn't have anaconda python in his/her device
1) Install miniconda for Python 3.8 using the instruction (system-based) in the link below:

   https://docs.conda.io/en/latest/miniconda.html

2) Open termainal (Mac, Linux) or anaconda prompt (Miniconda3) as administrator (windows)

3) Move to the directory containing the unziped project files using 'cd' command

4) Create an new virtual environment with the following command:

   >conda create -n chatbot_api python=3.8.5

5) Activate the environment with the following command:

   >conda activate chatbot_api

6) Install the needed python packages with the following command:

   >pip install -r requirements.txt

7) Download needed nltk files by the following commands:
   >python
   
   >import nltk 
   
   >nltk.download()
 
 In the GUI window that opens simply press the 'Download' button to download all corpora

Now the user is set to run the codes of the projects

## Run
to the run the codes, the user 
