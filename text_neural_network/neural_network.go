package main

import (
	"flag"
	"fmt"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"

	"text_neural_network/functions"
)

var training_data map[string][]string
var training *mat.Dense
var words []string
var categories []string
var output *mat.Dense
var hidden_neurons = 20
var alpha = 0.1
var epochs = 100000
var dropout = false
var dropout_percent = 0.2
var details = false

func main() {
	//Separate file in lines
	line, err := functions.ScanPhrases("./chatss.txt")
	if err != nil {
		panic(err)
	}
	//Get the training data, words database and categroies database from our lines database
	training_data, words, categories = functions.SetDb(line)
	//Get the corresponding binary matrix of every sentences database, and categories database
	training, output = functions.Binarize(training_data, words, categories)
	//Set flag to be able to decide from cmd, train or test
	command := flag.String("command", "test", "Either train or test to evaluate neural network")
	//Set flag to be able to set a test sentence from cmd
	user_input := flag.String("user_input", "lloro", "Type a sentence for the chat bot")
	flag.Parse()

	// train the network or test to determine the effectiveness of the trained network
	switch *command {
	case "train":
		//Start time
		rand.Seed(time.Now().UTC().UnixNano())
		t1 := time.Now()
		//Train the database
		functions.Train(training, output, hidden_neurons, alpha, epochs, dropout, dropout_percent, words, categories)
		//End time
		elapsed := time.Since(t1)
		fmt.Printf("\nTime taken to train: %s\n", elapsed)
	case "test":
		//Load synapses, word database and categories database
		synapse_0, synapse_1, words, categories := functions.LoadFile("model.json")
		//Classify user input from cmd
		functions.Classify(*user_input, details, synapse_0, synapse_1, words, categories)
	default:
		// don't do anything
	}
}
