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
	var data Outmost
	err = json.Unmarshal([]byte(byteValue), &data)
	if err != nil {
		panic(err)
	}
	return data
}

func Train(x *mat.Dense, y *mat.Dense, hidden int, alpha float64, epochs int, dropout bool, dropout_percent float64, words_db []string, categories []string) {
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
	//Set random weights
	synapse_0 := mat.NewDense(c1, hidden, data1)
	synapse_1 := mat.NewDense(hidden, c2, data2)
	prev_synapse_0_weight_update := new(mat.Dense)
	prev_synapse_0_weight_update.CloneFrom(synapse_0)
	prev_synapse_0_weight_update.Zero()

	prev_synapse_1_weight_update := new(mat.Dense)
	prev_synapse_1_weight_update.CloneFrom(synapse_1)
	prev_synapse_1_weight_update.Zero()

	synapse_0_direction_count := new(mat.Dense)
	synapse_0_direction_count.CloneFrom(synapse_0)
	synapse_0_direction_count.Zero()

	synapse_1_direction_count := new(mat.Dense)
	synapse_1_direction_count.CloneFrom(synapse_1)
	synapse_1_direction_count.Zero()

	for ep := 0; ep <= epochs; ep++ {
		//Feed forward through layer 0, layer 1, and layer 2
		layer_0 := x
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

		layer_2_prod := new(mat.Dense)
		layer_2_prod.Product(layer_1, synapse_1)
		layer_2 := sigmoid(layer_2_prod)

		layer_2_error := new(mat.Dense)
		layer_2_error.Sub(y, layer_2)

		if (ep%10000) == 0 && ep > 5000 {
			//Check error
			var mean_err float64
			mean_err = mat_mean(absolute(layer_2_error))
			if mean_err < last_mean_error {
				fmt.Printf("delta after %v, iterations: %f\n", ep, mean_err)
				last_mean_error = mean_err
			} else {
				fmt.Printf("break, delta: %f > last_mean_error: %f\n", mean_err, last_mean_error)
				break
			}
		}
		layer_2_delta := new(mat.Dense)
		layer_2_delta.MulElem(layer_2_error, sig_to_dev(layer_2))

		layer_1_error := new(mat.Dense)
		layer_1_error.Product(layer_2_delta, synapse_1.T())

		layer_1_delta := new(mat.Dense)
		layer_1_delta.MulElem(layer_1_error, sig_to_dev(layer_1))

		synapse_1_weight_update := new(mat.Dense)
		synapse_0_weight_update := new(mat.Dense)
		synapse_1_weight_update.Product(layer_1.T(), layer_2_delta)
		synapse_0_weight_update.Product(layer_0.T(), layer_1_delta)

		if ep > 0 {
			synapse_0_binary := new(mat.Dense)
			prev_synapse_0_binary := new(mat.Dense)
			synapse_0_binary.Apply(func(i, j int, v float64) float64 {
				if v > 0 {
					v = 1
					return v
				} else {
					v = 0
					return v
				}
			}, synapse_0_weight_update)
			prev_synapse_0_binary.Apply(func(i, j int, v float64) float64 {
				if v > 0 {
					v = 1
					return v
				} else {
					v = 0
					return v
				}
			}, prev_synapse_0_weight_update)
			sub_0_result := new(mat.Dense)
			sub_0_result.Sub(synapse_0_binary, prev_synapse_0_binary)
			synapse_0_direction_count.Add(synapse_0_direction_count, absolute(sub_0_result))

			synapse_1_binary := new(mat.Dense)
			prev_synapse_1_binary := new(mat.Dense)
			synapse_1_binary.Apply(func(i, j int, v float64) float64 {
				if v > 0 {
					v = 1
					return v
				} else {
					v = 0
					return v
				}
			}, synapse_1_weight_update)
			prev_synapse_1_binary.Apply(func(i, j int, v float64) float64 {
				if v > 0 {
					v = 1
					return v
				} else {
					v = 0
					return v
				}
			}, prev_synapse_1_weight_update)
			sub_1_result := new(mat.Dense)
			sub_1_result.Sub(synapse_1_binary, prev_synapse_1_binary)
			synapse_1_direction_count.Add(synapse_1_direction_count, absolute(sub_1_result))

		}
		mul := new(mat.Dense)
		mul.Apply(func(i, j int, v float64) float64 { return alpha * v }, synapse_1_weight_update)
		synapse_1.Add(synapse_1, mul)

		mul1 := new(mat.Dense)
		mul1.Apply(func(i, j int, v float64) float64 { return alpha * v }, synapse_0_weight_update)
		synapse_0.Add(synapse_0, mul1)

		prev_synapse_0_weight_update.CloneFrom(synapse_0_weight_update)
		prev_synapse_1_weight_update.CloneFrom(synapse_1_weight_update)

	}
	//Persist Synapses
	data := synapse{
		Synapse_0:  synapse_0.RawMatrix(),
		Synapse_1:  synapse_1.RawMatrix(),
		Words:      words_db,
		Categories: categories,
	}
	file, err := json.Marshal(data)
	if err != nil {
		fmt.Println(err)
	}
	_ = ioutil.WriteFile("model.json", file, 0644)
}

func Classify(sentence string, details bool, synapse_0 *mat.Dense, synapse_1 *mat.Dense, words []string, categories []string) Entries {
	var result *mat.Dense
	result = think(sentence, details, synapse_0, synapse_1, words)
	_, c := result.Dims()
	prediction := make(map[int]float64)
	for i := 0; i < c; i++ {
		if result.At(0, i) > ERROR_THRESHOLD {
			prediction[i] = result.At(0, i)
		}
	}
	var es Entries
	for k, v := range prediction {
		es = append(es, Entry{Val: v, Key: categories[k]})
	}

	sort.Sort(sort.Reverse(es))
	fmt.Printf("Input: %s\n Category: %v Confidence: %v\n", sentence, es[0].Key, es[0].Val)
	answer := response(es)
	return answer
}

func think(sentence string, details bool, synapse_0 *mat.Dense, synapse_1 *mat.Dense, words []string) *mat.Dense {
	x := bow(sentence, words, details)
	if details {
		fmt.Println("sentence:", sentence, "\nbow:", x)
	}
	//Input Layer
	l0 := x
	//Matrix multiplication Intput and Hidden
	d := new(mat.Dense)
	d.Product(l0, synapse_0)
	l1 := sigmoid(d)
	d1 := new(mat.Dense)
	d1.Product(l1, synapse_1)
	l2 := sigmoid(d1)

	return l2
}

func PhraseSplitFunc(data []byte, atEOF bool) (advance int, token []byte, err error) {

	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}

	if i := strings.Index(string(data), ")"); i >= 0 {
		if string(data[0:i]) != "" {
			return i + 1, data[0 : i+1], nil
		}
	}

	if atEOF {
		return len(data), data, nil
	}

	return
}

func ScanPhrases(path string) ([]string, error) {

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	scanner := bufio.NewScanner(file)

	scanner.Split(PhraseSplitFunc)

	var phrase []string

	for scanner.Scan() {
		phrase = append(phrase, scanner.Text())
	}
	return phrase, err
}

func scanWords(list string) []string {

	scanner := bufio.NewScanner(strings.NewReader(list))

	scanner.Split(bufio.ScanWords)
	var boolean bool
	var words []string
	ign := []string{"la", "a", "un", "una", "?", "!", "el", "con", "sin", "en", "para",
		"por", ".", "siempre", "desde", "los", "las", "me", "que", "tan", "de", "favor"}

	for scanner.Scan() {
		text := scanner.Text()
		boolean, _ = Find(ign, strings.ToLower(text))
		if boolean {
			continue
		} else {
			t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
			result, _, _ := transform.String(t, text)
			words = append(words, strings.ToLower(result))
		}
	}

	return words
}

func SetDb(line []string, db map[string][]string) map[string][]string {

	re1 := regexp.MustCompile(`\((.+)\)`)
	re2 := regexp.MustCompile(`\#(.+)\s\(`)
	db = make(map[string][]string)
	for _, word := range line {
		id := string(re1.FindAllSubmatch([]byte(word), -1)[0][1])
		if id != "noanswer" {
			text := string(re2.FindAllSubmatch([]byte(word), -1)[0][1])
			db[id] = append(db[id], text)
		} else {
			db[id] = append(db[id], "")

		}

	}
	return db
}

func Organize(db map[string][]string) ([]string, []string) {
	var boolean bool
	count := 0
	keys := []string{}

	for k, _ := range db {
		count += len(db[k])
		keys = append(keys, k)
	}
	sort.Strings(keys)
	words := []string{}
	categories := []string{}

	for _, k := range keys {
		for _, item := range db[k] {
			w := scanWords(item)
			for _, wrd := range w {
				boolean, _ = Find(words, wrd)
				if !boolean {
					words = append(words, wrd)
				}
			}
		}
		boolean, _ = Find(categories, k)
		if !boolean {
			categories = append(categories, k)
		}
	}
	return words, categories
}

func Find(slice []string, val string) (bool, int) {
	for i, _ := range slice {
		if slice[i] == val {
			return true, i
		}
	}
	return false, 0
}

func Binarize(db map[string][]string, word_db []string, categories []string) (*mat.Dense, *mat.Dense) {

	count := 0
	keys := []string{}

	for k, _ := range db {
		count += len(db[k])
		keys = append(keys, k)
	}
	sort.Strings(keys)

	training := mat.NewDense(count, len(word_db), nil)
	output := mat.NewDense(count, len(categories), nil)

	i := 0
	for _, k := range keys {
		pattern_words := []string{}
		for _, word := range db[k] {
			pattern_words = scanWords(word)
			for _, w := range pattern_words {
				for j, item := range word_db {
					if item == w {
						training.Set(i, j, 1)
					}
				}
			}
			for q, item := range categories {
				if item == k {
					output.Set(i, q, 1)
				}
			}
			i++
		}
	}
	return training, output
}

func response(category Entries) Entries {
	var sentence string
	var v int
	intents_db := LoadIntens("C:\\Users\\jrtor\\go\\src\\text_neural_network\\intents.json")

	switch category[0].Key {
	case "greeting":
		v = rand.Intn(len(intents_db.Category.Greeting))
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

	var es Entries
	es = append(es, Entry{Val: category[0].Val, Key: sentence})
	return es
}

func sigmoid(v *mat.Dense) *mat.Dense {
	r, c := v.Dims()
	output := mat.NewDense(r, c, nil)
	output.Apply(sig, v)
	return output
}
func sig(i, j int, v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-1*v))
}

func sig_to_dev(v *mat.Dense) *mat.Dense {
	m := new(mat.Dense)
	output := new(mat.Dense)
	m.Apply(func(i, j int, v float64) float64 { return (1 - v) }, v)
	output.MulElem(v, m)
	return output
}

func clean_up_sentence(sentence string) []string {
	sentence_words := scanWords(sentence)
	return sentence_words
}

func bow(sentence string, words_db []string, details bool) *mat.Dense {
	sentence_words := clean_up_sentence(sentence)
	bag := make([]float64, len(words_db))
	for _, s := range sentence_words {
		for i, item := range words_db {
			if item == s {
				bag[i] = 1
			}
		}
	}
	if details {
		fmt.Println(bag)
	}
	v := mat.NewDense(1, len(bag), bag)
	return v
}

func absolute(m1 *mat.Dense) *mat.Dense {
	m_abs := new(mat.Dense)
	m_abs.Apply(func(i, j int, v float64) float64 { return math.Abs(v) }, m1)
	return m_abs
}

func mat_mean(m1 *mat.Dense) float64 {
	r, c := m1.Dims()
	var val float64
	for i := 0; i < r-1; i++ {
		for j := 0; j < c-1; j++ {
			val += m1.At(i, j)
		}
	}
	mean := val / float64(r*c)
	return mean
}

//func word_checker(word string, db_word string) float64 {
//	count := 0
//		for _, glet := range db_word {
//			if ulet == glet {
//			} else {
//			}
//		}
//
//	}
//	similarity := float64(count) / float64(len(db_word))
//	return similarity
//}

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
