package functions

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/mat"
)

var ERROR_THRESHOLD = 0.2

func LoadFile(file string) (*mat.Dense, *mat.Dense, []string, []string) {
	// load our calculated synapse values
	jsonFile, err := os.Open(file)
	// if we os.Open returns an error then handle it
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened model.json")
	defer jsonFile.Close()

	//read our opened json file
	byteValue, _ := ioutil.ReadAll(jsonFile)
	//Initialize map
	var data synapse
	err = json.Unmarshal([]byte(byteValue), &data)
	if err != nil {
		panic(err)
	}

	s_0 := data.Synapse_0
	s_1 := data.Synapse_1
	synapse_0 := mat.NewDense(s_0.Rows, s_0.Cols, s_0.Data)
	synapse_1 := mat.NewDense(s_1.Rows, s_1.Cols, s_1.Data)
	words := data.Words
	categories := data.Categories

	return synapse_0, synapse_1, words, categories
}

func LoadIntens(file string) Outmost {
	// load our intents file
	jsonFile, err := os.Open(file)
	// if we os.Open returns an error then handle it
	if err != nil {
		fmt.Println(err)
	}
	fmt.Println("Successfully Opened model.json")
	defer jsonFile.Close()

	//read our opened json file
	byteValue, _ := ioutil.ReadAll(jsonFile)
	//Initialize data for storing the intents
	var data Outmost
	err = json.Unmarshal([]byte(byteValue), &data)
	if err != nil {
		panic(err)
	}
	return data
}

func Train(x *mat.Dense, y *mat.Dense, hidden int, alpha float64, epochs int, dropout bool, dropout_percent float64, words_db []string, categories []string) {
	//We get the dimension of input x and y
	rx, cx := x.Dims()
	ry, cy := y.Dims()
	fmt.Printf("Training with %v,  alpha: %f, dropout: %t\n", hidden, alpha, dropout)
	fmt.Printf("Input matrix: %vx%v  Output matrix: %vx%v\n", rx, cx, ry, cy)

	last_mean_error := float64(1)
	//Randomly set data for the weights x
	_, c1 := x.Dims()
	data1 := make([]float64, c1*hidden)
	for i := range data1 {
		data1[i] = 2*rand.Float64() - 1
	}
	//Randomly set data for the weights y
	_, c2 := y.Dims()
	data2 := make([]float64, hidden*c2)
	for i := range data2 {
		data2[i] = 2*rand.Float64() - 1
	}
	//Set random weights for synapse_1 and synapse_0
	synapse_0 := mat.NewDense(c1, hidden, data1)
	synapse_1 := mat.NewDense(hidden, c2, data2)

	for ep := 0; ep <= epochs; ep++ {
		//Set input, layer 0
		layer_0 := x
		//Feed forward to layer 1
		layer_0_prod := new(mat.Dense)
		layer_0_prod.Product(layer_0, synapse_0)
		layer_1 := sigmoid(layer_0_prod)

		if dropout {
			data3 := make([]float64, c1*hidden)
			for i := range data3 {
				r := rand.Float64()
				data3[i] = r / r
			}
			ones := mat.NewDense(c1, hidden, data3)
			layer_1.Product(layer_1, ones)
		}
		//Forward propagation to layer 2
		layer_2_prod := new(mat.Dense)
		layer_2_prod.Product(layer_1, synapse_1)
		layer_2 := sigmoid(layer_2_prod)

		//Get error of layer_2
		layer_2_error := new(mat.Dense)
		layer_2_error.Sub(y, layer_2)

		if (ep%10000) == 0 && ep > 5000 {
			//Check error
			var mean_err float64
			mean_err = mat_mean(absolute(layer_2_error))
			//If error is decreasing all GOOD, continue
			if mean_err < last_mean_error {
				fmt.Printf("delta after %v, iterations: %f\n", ep, mean_err)
				last_mean_error = mean_err
			} else {
				//If error is getting bigger then something is wrong, stop the process
				fmt.Printf("break, delta: %f > last_mean_error: %f\n", mean_err, last_mean_error)
				break
			}
		}
		//Backward propagation layer 2, derivative and error
		layer_2_delta := new(mat.Dense)
		layer_2_delta.MulElem(layer_2_error, sig_to_dev(layer_2))

		//Backward propagation layer 1, derivative and error
		layer_1_error := new(mat.Dense)
		layer_1_error.Product(layer_2_delta, synapse_1.T())

		//Get delta of layer_1 from error
		layer_1_delta := new(mat.Dense)
		layer_1_delta.MulElem(layer_1_error, sig_to_dev(layer_1))

		//Get the updated weights based on gradient decent
		synapse_1_weight_update := new(mat.Dense)
		synapse_0_weight_update := new(mat.Dense)
		synapse_1_weight_update.Product(layer_1.T(), layer_2_delta)
		synapse_0_weight_update.Product(layer_0.T(), layer_1_delta)

		//Apply the learning rate alpha to synapse 1 matrix
		mul := new(mat.Dense)
		mul.Apply(func(i, j int, v float64) float64 { return alpha * v }, synapse_1_weight_update)
		synapse_1.Add(synapse_1, mul)

		//Aply learning rate alpha to synapse 0 matrix
		mul1 := new(mat.Dense)
		mul1.Apply(func(i, j int, v float64) float64 { return alpha * v }, synapse_0_weight_update)
		synapse_0.Add(synapse_0, mul1)

	}
	//Store the synapse matrixes values on data
	data := synapse{
		Synapse_0:  synapse_0.RawMatrix(),
		Synapse_1:  synapse_1.RawMatrix(),
		Words:      words_db,
		Categories: categories,
	}
	//Encode the data into a json
	file, err := json.Marshal(data)
	if err != nil {
		fmt.Println(err)
	}
	//Save the file into model.json
	_ = ioutil.WriteFile("model.json", file, 0644)
}

func Classify(sentence string, details bool, synapse_0 *mat.Dense, synapse_1 *mat.Dense, words []string, categories []string) Entries {
	var result *mat.Dense
	//Get the prediction of the ANN, and save it
	result = think(sentence, details, synapse_0, synapse_1, words)
	//Get the columns of result
	_, c := result.Dims()
	prediction := make(map[int]float64)
	//Iterate thorugh all categories and identify the ones greater than ERROR_THRESHOLD
	for i := 0; i < c; i++ {
		if result.At(0, i) > ERROR_THRESHOLD {
			//Save the ones that have a high certainity value
			prediction[i] = result.At(0, i)
		}
	}
	var es Entries
	//Iterate through the map to identify the value and the index of the prediciton
	for k, v := range prediction {
		//Get the corresponding category from the categoies array
		es = append(es, Entry{Val: v, Key: categories[k]})
	}
	if es != nil {
		fmt.Printf("Input: %s\n Category: %v Confidence: %v\n", sentence, es[0].Key, es[0].Val)
	} else {
		es = append(es, Entry{Val: 99.99, Key: categories[10]})
		fmt.Printf("Input: %s\n Category: %v Confidence: %v\n", sentence, es[0].Key, es[0].Val)
	}
	//Get the response based on the identified category
	answer := response(es)
	fmt.Printf("Output: %v\n", answer[0].Key)
	return answer
}

func think(sentence string, details bool, synapse_0 *mat.Dense, synapse_1 *mat.Dense, words []string) *mat.Dense {
	//Given a sentence, get the binary vector according to the words used, and the word on the data base
	x := bow(sentence, words, details)
	if details {
		fmt.Println("sentence:", sentence, "\nbow:", x)
	}
	//Input the binarized sentence as fisrt Layer
	l0 := x
	//Matrix multiplication Intput and Hidden layer
	d := new(mat.Dense)
	d.Product(l0, synapse_0)
	l1 := sigmoid(d)
	d1 := new(mat.Dense)
	d1.Product(l1, synapse_1)
	//Output layer, response of the newtwork
	l2 := sigmoid(d1)
	return l2
}

//This function is going to set the split rule, to ")"
func PhraseSplitFunc(data []byte, atEOF bool) (advance int, token []byte, err error) {
	//If we are at the end of file and there no  data
	if atEOF && len(data) == 0 {
		//Then we finished
		return 0, nil, nil
	}
	//We search for ")" on the data, if it is after the first character of data
	if i := strings.Index(string(data), ")"); i >= 0 {
		//Then we get all the characters before ") and if this is not empty"
		if string(data[0:i]) != "" {
			//We then move on after the ")" and return all the data before ")"
			return i + 1, data[0 : i+1], nil
		}
	}
	//If we are at end of file
	if atEOF {
		//We return the remain data, and the index after this data
		return len(data), data, nil
	}

	return
}

//This function split our file in lines of sentences,category
func ScanPhrases(path string) ([]string, error) {
	//We open the database file
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	//No matter what we close it
	defer file.Close()
	//We start a new sacnner of our database file
	scanner := bufio.NewScanner(file)
	//We split our file in lines according to the custom PhraseSplitFunction
	scanner.Split(PhraseSplitFunc)

	var phrase []string
	//We scan all inside the file
	for scanner.Scan() {
		//Store each phrase and category inside a slice
		phrase = append(phrase, scanner.Text())
	}
	return phrase, err
}

//This function is going get every word remove accents and remove articles and unnecesary words
func scanWords(list string) []string {
	//We start a new scanner
	scanner := bufio.NewScanner(strings.NewReader(list))
	//We split according to spaces
	scanner.Split(bufio.ScanWords)
	var boolean bool
	var words []string
	ign := []string{"la", "a", "un", "una", "?", "!", "el", "con", "sin", "en", "para",
		"por", ".", "siempre", "desde", "los", "las", "me", "que", "tan", "de", "favor"}
	//We scan for every word
	for scanner.Scan() {
		text := scanner.Text()
		//We find if the word is inside ignored database
		boolean, _ = Find(ign, strings.ToLower(text))
		if boolean {
			//If true then we ignore it
			continue
		} else {
			//If it is not, then we remove accents
			t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
			result, _, _ := transform.String(t, text)
			//Save it on the word data base
			words = append(words, strings.ToLower(result))
		}
	}

	return words
}

//This function set our data base correcly, on a map in the way category:sentences
//And get a word_database of al unique words of all sentences
//And a category database of all categories in the database
func SetDb(line []string) (map[string][]string, []string, []string) {
	words := []string{}
	categories := []string{}
	keys := []string{}
	var id, text string
	var boolean bool
	//Set the REGEX rules for category
	re1 := regexp.MustCompile(`\((.+)\)`)
	//Set the regex rule for sentence
	re2 := regexp.MustCompile(`\#(.+)\s\(`)
	//Initialize database map
	db := make(map[string][]string)
	//Iterate through every word in the line
	for _, sentence := range line {
		//Identify the category
		id = string(re1.FindAllSubmatch([]byte(sentence), -1)[0][1])
		if id != "noanswer" {
			//Identify the sentence
			text = string(re2.FindAllSubmatch([]byte(sentence), -1)[0][1])
			//Save the category and sentence
			db[id] = append(db[id], text)
			//If the category is "noanswer" just insert an "" as sentence
		} else {
			db[id] = append(db[id], "")
		}
		//Get an array of every word on the sentence
		w := scanWords(text)
		//Iterate through all words of the sentence
		for _, wrd := range w {
			//Find if wrd is already in words
			boolean, _ = Find(words, wrd)
			if !boolean {
				//If wrd not in words then add it
				words = append(words, wrd)
			}
		}
	}
	//Get an array with all categories
	for k, _ := range db {
		keys = append(keys, k)
	}
	//Sort keys on alphabetical order
	sort.Strings(keys)
	for _, k := range keys {
		//Assign categories in alphabetical order to variable categories
		categories = append(categories, k)
	}

	return db, words, categories
}

//This function finds if a string is already present on a slice
func Find(slice []string, val string) (bool, int) {
	//Iterate throug index of a slice
	for i, _ := range slice {
		//Compare if the string on i index is equal to val string
		if slice[i] == val {
			return true, i
		}
	}
	return false, 0
}

//This function change our sentences database into a binary array of sentences
func Binarize(db map[string][]string, word_db []string, categories []string) (*mat.Dense, *mat.Dense) {

	count := 0
	keys := []string{}
	//We get an array of keys of the database map
	for k, _ := range db {
		count += len(db[k])
		keys = append(keys, k)
	}
	//We sort them in alphabetical order
	sort.Strings(keys)

	//Initialize zero matrix of sentences(training) and categories(output)
	training := mat.NewDense(count, len(word_db), nil)
	output := mat.NewDense(count, len(categories), nil)

	i := 0
	//Iteration through every category
	for _, k := range keys {
		pattern_words := []string{}
		//Iteration through every sentence
		for _, sentence := range db[k] {
			//Get all words in the sentence
			pattern_words = scanWords(sentence)
			//For every word in slice of words
			for _, w := range pattern_words {
				//For every word in word database
				for j, item := range word_db {
					//If both are the same
					if item == w {
						//Assign a one
						training.Set(i, j, 1)
					}
				}
			}
			//For every category
			for q, item := range categories {
				//If they are equal
				if item == k {
					//Assign a one
					output.Set(i, q, 1)
				}
			}
			//Move to the next row on the matrix
			i++
		}
	}
	return training, output
}

func response(category Entries) Entries {
	var sentence string
	var v int
	//Load intents from file
	intents_db := LoadIntens("C:\\Users\\jrtor\\go\\src\\text_neural_network\\intents.json")
	//Search for the correct category
	switch category[0].Key {
	case "greeting":
		//Generate a random number
		v = rand.Intn(len(intents_db.Category.Greeting))
		//Choose a phrase of index v from array
		sentence = intents_db.Category.Greeting[v]
	case "goodbye":
		v = rand.Intn(len(intents_db.Category.Goodbye))
		sentence = intents_db.Category.Goodbye[v]
	case "thanks":
		v = rand.Intn(len(intents_db.Category.Thanks))
		sentence = intents_db.Category.Thanks[v]
	case "noanswer":
		v = rand.Intn(len(intents_db.Category.Noanswer))
		sentence = intents_db.Category.Noanswer[v]
	case "options":
		v = rand.Intn(len(intents_db.Category.Options))
		sentence = intents_db.Category.Options[v]
	case "food,order,pizza":
		v = rand.Intn(len(intents_db.Category.Orderpizza))
		sentence = intents_db.Category.Orderpizza[v]
	case "food,order,hamburger":
		v = rand.Intn(len(intents_db.Category.Orderham))
		sentence = intents_db.Category.Orderham[v]
	case "food,order,salad":
		v = rand.Intn(len(intents_db.Category.Ordersalad))
		sentence = intents_db.Category.Ordersalad[v]
	case "drinks,order,water":
		v = rand.Intn(len(intents_db.Category.Orderwater))
		sentence = intents_db.Category.Orderwater[v]
	case "drinks,order,tea":
		v = rand.Intn(len(intents_db.Category.Ordertea))
		sentence = intents_db.Category.Ordertea[v]
	case "drinks,order,soda":
		v = rand.Intn(len(intents_db.Category.Ordersoda))
		sentence = intents_db.Category.Ordersoda[v]
	case "disliked":
		v = rand.Intn(len(intents_db.Category.Disliked))
		sentence = intents_db.Category.Disliked[v]
	case "liked":
		v = rand.Intn(len(intents_db.Category.Liked))
		sentence = intents_db.Category.Liked[v]
	}
	//Save sentence inside es, with actual value of centainty
	var es Entries
	es = append(es, Entry{Val: category[0].Val, Key: sentence})
	return es
}

//This function applies the function sigmoid elemnt wise to a matrix
func sigmoid(v *mat.Dense) *mat.Dense {
	//Get matrix dimension
	r, c := v.Dims()
	//Initialize matrix output
	output := mat.NewDense(r, c, nil)
	//Apply sigmoid function to every element of v
	output.Apply(func(i, j int, v float64) float64 { return 1.0 / (1.0 + math.Exp(-1*v)) }, v)
	return output
}

//This function applies the second derivate of sigmoid function to matrix v
func sig_to_dev(v *mat.Dense) *mat.Dense {
	//Initialize matrix m
	m := new(mat.Dense)
	//Initialize matrix output
	output := new(mat.Dense)
	//Apply "1-v" element wise to every element in v
	m.Apply(func(i, j int, v float64) float64 { return (1 - v) }, v)
	//Multiply element wise every element of v and m
	output.MulElem(v, m)
	return output
}

//This function binarize the input sentence to be able to insert it to the NN
func bow(sentence string, words_db []string, details bool) *mat.Dense {
	//Get every word of the sentence
	sentence_words := scanWords(sentence)
	//Initialize slice of float64
	bag := make([]float64, len(words_db))
	//Iterate through every word of the sentence
	for _, word := range sentence_words {
		//Iterate through every word in words database
		for i, item := range words_db {
			//If they are equal
			if item == word {
				//Set a one
				bag[i] = 1
			}
		}
	}
	if details {
		fmt.Println(bag)
	}
	//Create new vector from slice
	v := mat.NewDense(1, len(bag), bag)
	return v
}

//This function gets the absolute value of a matrix
func absolute(m1 *mat.Dense) *mat.Dense {
	//Initialize matrix
	m_abs := new(mat.Dense)
	//Apply math.abs to every element in m1
	m_abs.Apply(func(i, j int, v float64) float64 { return math.Abs(v) }, m1)
	return m_abs
}

//This function gets the mean of a matrix
func mat_mean(m1 *mat.Dense) float64 {
	//Get dimensions of m1
	r, c := m1.Dims()
	var val float64
	//Iterate through every element in the matrix
	for i := 0; i < r-1; i++ {
		for j := 0; j < c-1; j++ {
			//Add every value
			val += m1.At(i, j)
		}
	}
	//Divide total sum and total numbers
	mean := val / float64(r*c)
	return mean
}

type Entry struct {
	Val float64
	Key string
}

type Entries []Entry

func (s Entries) Len() int           { return len(s) }
func (s Entries) Less(i, j int) bool { return s[i].Val < s[j].Val }
func (s Entries) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func isMn(r rune) bool {
	return unicode.Is(unicode.Mn, r) // Mn: nonspacing marks
}

type synapse struct {
	Synapse_0  blas64.General
	Synapse_1  blas64.General
	Words      []string
	Categories []string
}

type Outmost struct {
	Category Inner
}

type Inner struct {
	Greeting   []string
	Goodbye    []string
	Thanks     []string
	Noanswer   []string
	Options    []string
	Orderpizza []string
	Orderham   []string
	Ordersalad []string
	Orderwater []string
	Ordertea   []string
	Ordersoda  []string
	Disliked   []string
	Liked      []string
}
