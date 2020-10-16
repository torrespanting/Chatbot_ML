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
	//Separamos el archivo de texto en lineas
	line, err := functions.ScanPhrases("./chatss.txt")
	if err != nil {
		panic(err)
	}
	//Creamos variable de base de datos
	//Analizamos cada linea del texto separada, para dientificar el mensaje y la clasificacion
	training_data = functions.SetDb(line, training_data)

	var words []string
	var categories []string
	words, categories = functions.Organize(training_data)

	training, output = functions.Binarize(training_data, words, categories)

	command := flag.String("command", "train", "Either train or test to evaluate neural network")
	user_input := flag.String("user_input", "quiero una pizza", "Type a sentence for the chat bot")
	flag.Parse()

	// train or  predict to determine the effectiveness of the trained network
	switch *command {
	case "train":
		rand.Seed(time.Now().UTC().UnixNano())
		t1 := time.Now()
		functions.Train(training, output, hidden_neurons, alpha, epochs, dropout, dropout_percent, words, categories)
		elapsed := time.Since(t1)
		fmt.Printf("\nTime taken to train: %s\n", elapsed)
	case "test":
		synapse_0, synapse_1, words, categories := functions.LoadFile("model.json")
		functions.Classify(*user_input, details, synapse_0, synapse_1, words, categories)
	default:
		// don't do anything
	}
}
