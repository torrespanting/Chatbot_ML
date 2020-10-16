package handlers

import (
	"encoding/json"
	"net/http"
	"text_neural_network/functions"
)

var detail bool

//A handler to fetch all the jobs
func GetResponse(w http.ResponseWriter, r *http.Request) {
	//make a slice to hold our jobs data
	synapse_0, synapse_1, words, categories := functions.LoadFile("C:\\Users\\jrtor\\go\\src\\text_neural_network\\model.json")
	val := r.FormValue("msg")
	category := functions.Classify(val, detail, synapse_0, synapse_1, words, categories)

	w.Header().Set("Content-Type", "application/json")
	for _, items := range category {
		json.NewEncoder(w).Encode(items)
	}
}
