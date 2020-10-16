# Chatbot_ML 
## This is a Restaurant Chatbot for helping on different task, like greeting users, take order on different food/drinks,among other functions
#### In the following line I'm going to describe how to train and test the chatbot on the UI webpage.
#### 1. Fisrt *copy* or *download* the repo
#### *__Note:__* Be sure to save it on your *GOPATH*, inside *src* folder, or else you will have problems when looking for functions in other packages
#### 2. One you have all the files you can run the file *main.go*, inside *__web_api__*
#### 3. This file will initialize the server, serve the html file on port *_3000_*, you can easily change this inside *_main.go_* file
#### 4. Once is loaded, you can co to *_localhost:3000_*, and insert a user
#### 5. And that's it, you can now star chatting with the bot

## Neural Network
#### If you want to modify the data base of the bot, you have to edit the *_chatss.txt_* file inisde the *_text_neural_network_* folder. there you will find the following structure: #*_sentence_* *_(_*category*_)_*, be sure to follow this format, as it was taken as a directive to set the database of the network.
#### If you want to add new categories, be sure to add some examples to the *_chatss.txt_*, and add the respective responses inside *_intents.json_*
#### For the network to work after modifications, you need to re-train it, be sure to follow the following steps:
#### 1. Navigate to the *_text_neural_network_* folder
#### 1. Build the *_neural_network.go_* fiel with : *go build text_neural_network*
#### 2. To train the neural netwokr being inside the *_neural_network_* folder
#### 3. Execture the following to build the .exe file: *_go build text_neural_network_*
#### 4. After building it, run the following for training: *_text_neural_network -command=train_*
#### 5. Then after finishing the training you can test an input if you want with: *_text_neural_network -command=test user_input="test_sentence_here"_*

## Final Comments
#### This is an early model, I'm currently workin on, I'm planning on keep doing improves to the code, and expanding the data base. Also implementing other features like, grammar mistakes identifier, and typo errors identification. 
#### If you have any commments or suggestion please, you are welcome to contact me.
