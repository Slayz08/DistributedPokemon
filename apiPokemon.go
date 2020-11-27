package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"sync"

	"github.com/go-gota/gota/series"
	"github.com/gorilla/mux"
	"github.com/kniren/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)

var wg sync.WaitGroup
var port string
var remotes []string

type matrix struct {
	dataframe.DataFrame
}

func (m matrix) At(i, j int) float64 {
	return m.Elem(i, j).Float()
}

func (m matrix) T() mat.Matrix {
	return mat.Transpose{m}
}

func sigmoidUtil(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func sigmoid(x mat.Matrix) mat.Matrix {
	evalMatrix := x
	outputs := mat.Col(nil, 0, evalMatrix) //change outputs

	size := len(outputs)
	processOutputs := make([]float64, size)
	for i, value := range outputs {
		processOutputs[i] = sigmoidUtil(value)
	}
	return mat.NewDense(size, 1, processOutputs)
}

func sumElements(x mat.Matrix) float64 {
	outputs := mat.Col(nil, 0, x)

	sum := 0.0
	for _, value := range outputs {
		sum += value
	}
	return sum
}

func zeros(n int) []float64 {
	a := make([]float64, n)
	for i := range a {
		a[i] = 0
	}
	return a
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func subtractScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return subtract(n, m)
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func multiplyScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return multiply(m, n)
}

func logMatrix(m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	util := mat.Col(nil, 0, m)
	a := make([]float64, r*c)
	for x, value := range util {
		a[x] = math.Log(value)
	}
	n := mat.NewDense(r, c, a)
	return n
}

func trainTestSplit(filename string, size int, xTrain chan mat.Matrix, xTest chan mat.Matrix, yTrain chan mat.Matrix, yTest chan mat.Matrix) {

	xSize := (size * 80) / 100

	pokemonMatchupsTrain, err := os.Open(filename)
	pokemonMatchupsTest, err := os.Open(filename)

	if err != nil {
		log.Fatalln("No se puede abrir el archivo", err)
	}

	rand.Seed(42)
	//Read file
	file := dataframe.ReadCSV(pokemonMatchupsTrain)
	fileData := file.Select([]string{"Index", "Hp_1", "Attack_1", "Hp_2", "Attack_2"})

	fileData = fileData.Filter(dataframe.F{"Index", series.LessEq, xSize})
	fileData = fileData.Drop(0)

	//set x train data
	var train mat.Matrix
	train = matrix{fileData}

	//set test
	file2 := dataframe.ReadCSV(pokemonMatchupsTest)
	file2Data := file2.Select([]string{"Index", "Hp_1", "Attack_1", "Hp_2", "Attack_2"})

	file2Data = file2Data.Filter(dataframe.F{"Index", series.Greater, xSize})
	file2Data = file2Data.Drop(0)

	//set y train data
	var test mat.Matrix
	test = matrix{file2Data}

	file3Data := file2.Select([]string{"Index", "Winner"})
	file3Data = file3Data.Filter(dataframe.F{"Index", series.LessEq, xSize})
	file3Data = file3Data.Drop(0)

	//set x test data
	var train2 mat.Matrix
	train2 = matrix{file3Data}

	//set y test data
	file4Data := file2.Select([]string{"Index", "Winner"})
	file4Data = file4Data.Filter(dataframe.F{"Index", series.Greater, xSize})
	file4Data = file4Data.Drop(0)

	var test2 mat.Matrix
	test2 = matrix{file4Data}

	xTrain <- train
	xTest <- test
	yTrain <- train2
	yTest <- test2

	wg.Done()

}

func approximate(X mat.Matrix, weights mat.Matrix, bias float64, nRow int, nCol int) mat.Matrix {
	mResult := mat.NewDense(nRow, nCol, nil)
	mResult.Product(X, weights)
	linearModel := addScalar(bias, mResult)
	return linearModel
}

func computeGradients(matrixUtil2 mat.Matrix, X mat.Matrix, y mat.Matrix, nSamples int, nFeatures int) (mat.Matrix, float64) {
	yPredicted := matrixUtil2
	ySub := subtract(yPredicted, y)
	_, ySubCol := ySub.Dims()
	mProd := mat.NewDense(nFeatures, ySubCol, nil)
	mProd.Product(X.T(), ySub)
	//prueba = mProd
	mvar := 1.0 / float64(nSamples)
	dw := multiplyScalar(mvar, mProd)
	db := mvar * sumElements(ySub)
	return dw, db
}

type LogRegression struct {
	lr      float64
	nIters  int
	weights mat.Matrix
	bias    float64
}

func costFunction(predictions mat.Matrix, y mat.Matrix, costResult chan float64) {
	observations, _ := y.Dims()
	//For error when 1
	negY := multiplyScalar(-1.0, y)
	logPredictions := logMatrix(predictions)
	class1Cost := multiply(negY, logPredictions)

	//For error when 0
	compY := subtractScalar(1, y)
	logCompPredictions := logMatrix(subtractScalar(1, predictions))
	class2Cost := multiply(compY, logCompPredictions)

	//Take the sum
	costMat := subtract(class1Cost, class2Cost)
	cost := sumElements(costMat) / float64(observations)

	costResult <- cost
}

func countNonZero(x mat.Matrix) int {
	var nonZeros int
	values := mat.Col(nil, 0, x)
	for _, value := range values {
		if value != 0 {
			nonZeros += 1
		}
	}
	return nonZeros
}

func accuracyPred(yPredicted mat.Matrix, y mat.Matrix) float64 {
	nRow, _ := y.Dims()
	yResult := subtract(yPredicted, y)
	return (1.0 - float64(countNonZero(yResult))/float64(nRow)) * 100.0

}

func decisionBoundary(yPredicted mat.Matrix) mat.Matrix {
	r, c := yPredicted.Dims()
	values := mat.Col(nil, 0, yPredicted)
	a := make([]float64, r*c)
	for x, value := range values {
		if value < 0.5 {
			a[x] = 0.0
		} else {
			a[x] = 1.0
		}
	}
	yResult := mat.NewDense(len(a), 1, a)
	return yResult
}

func (l *LogRegression) fit(X mat.Matrix, y mat.Matrix) (mat.Matrix, float64) {
	//init parameters
	nSamples, nFeatures := X.Dims()
	_, lColumns := l.weights.Dims()
	costResult := make(chan float64)
	var prueba mat.Matrix
	//var cost float64
	var accuracy float64
	//gradient descent
	for i := 0; i < l.nIters-1; i++ {
		//approximate
		linearModel := approximate(X, l.weights, l.bias, nSamples, lColumns)
		//linearModel := <-matrix_util
		yPredicted := sigmoid(linearModel)
		yPredicted2 := decisionBoundary(yPredicted)
		prueba = yPredicted2
		accuracy = accuracyPred(yPredicted2, y)
		//compute gradients
		dw, db := computeGradients(yPredicted, X, y, nSamples, nFeatures)
		//update parameters
		l.weights = subtract(l.weights, multiplyScalar(l.lr, dw))
		l.bias -= l.lr * db
		//v1
		go costFunction(yPredicted, y, costResult)
		cost := <-costResult
		if i%1000 == 0 {
			fmt.Println("Iterador: ", i, "cost: ", cost)
		}
	}
	fmt.Println("Predictions: \n")
	matPrint(prueba)
	fmt.Println("Accuracy: ", accuracy, "%")
	return l.weights, l.bias
}

func (l *LogRegression) predict(X mat.Matrix, y mat.Matrix) (mat.Matrix, float64) {
	nSamples, _ := X.Dims()
	_, lColumns := l.weights.Dims()
	//matrix_predict := make(chan mat.Matrix)
	//matrix_result := make(chan mat.Matrix)
	linearModel := mat.NewDense(nSamples, lColumns, nil)
	linearModel.Product(X, l.weights)
	//matrix_predict <- linearModel
	yPredicted := decisionBoundary(sigmoid(linearModel))
	var accuracy float64
	if y != nil {
		accuracy = accuracyPred(yPredicted, y)
	} else {
		accuracy = 1.0
	}
	return yPredicted, accuracy

}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

//API CODE

type pokemon struct {
	ID      int    `json:ID`
	Name    string `json:Name`
	Type    string `json:Type`
	Hp      int    `json:Hp`
	Attack  int    `json:Attack`
	Defense int    `json:Defense`
	Speed   int    `json:Speed`
}

type election struct {
	winner int `json:winner`
}

type allPokemons []pokemon

var pokemons = allPokemons{
	{
		ID:      1,
		Name:    "Bulbasaur",
		Type:    "Grass",
		Hp:      45,
		Attack:  49,
		Defense: 49,
		Speed:   45,
	},
}

func getPokemons(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("content-type", "application/json")
	json.NewEncoder(w).Encode(pokemons)
}

func getBias(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("content-type", "application/json")
	json.NewEncoder(w).Encode(bot.bias)
}

func createPokemon(w http.ResponseWriter, r *http.Request) {
	var newTask pokemon

	requestBody, err := ioutil.ReadAll(r.Body)
	if err != nil {
		fmt.Fprintf(w, "Insert a Valid Task")
	}

	json.Unmarshal(requestBody, &newTask)
	requestBody2, _ := json.Marshal(newTask)

	newTask.ID = len(pokemons) + 1
	pokemons = append(pokemons, newTask)

	if port == "4000" {
		for _, p := range remotes {
			req, _ := http.NewRequest("POST", "http://localhost:"+p+"/pokemons", bytes.NewBuffer(requestBody2))
			req.Header.Set("content-type", "application/json")

			client := &http.Client{}
			client.Do(req)
		}
	}

	printPokemonBattle()

	w.Header().Set("content-type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(newTask)
}

func printPokemonBattle() {
	var ganador int

	fmt.Println("· Batalla Pokemon ·")
	fmt.Println("   Nombre   |  Tipo  | HP | Ataque | Defensa | Velocidad")
	fmt.Printf("1: %s | %s | %d | %d | %d | %d \n", pokemons[len(pokemons)-1].Name, pokemons[len(pokemons)-1].Type, pokemons[len(pokemons)-1].Hp, pokemons[len(pokemons)-1].Attack, pokemons[len(pokemons)-1].Defense, pokemons[len(pokemons)-1].Speed)
	fmt.Printf("2: %s | %s | %d | %d | %d | %d \n", pokemons[len(pokemons)-2].Name, pokemons[len(pokemons)-2].Type, pokemons[len(pokemons)-2].Hp, pokemons[len(pokemons)-2].Attack, pokemons[len(pokemons)-2].Defense, pokemons[len(pokemons)-2].Speed)
	fmt.Print("Elige al ganador de la batalla! (1/2): ")
	fmt.Scanf("%s", &ganador)

	//actualizar opcion y mostrar la de los otros

	for _, p := range remotes {
		election := election{winner: ganador}
		request, _ := json.Marshal(election)
		req, _ := http.NewRequest("POST", "http://localhost:"+p+"/elections", bytes.NewBuffer(request))
		req.Header.Set("content-type", "application/json")

		client := &http.Client{}
		client.Do(req)
	}

}

func postElection(w http.ResponseWriter, r *http.Request) {

}

func getPokemonWithID(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID, err := strconv.Atoi(vars["id"])
	if err != nil {
		fmt.Fprintf(w, "Invalid ID")
		return
	}

	for _, task := range pokemons {
		if task.ID == taskID {
			w.Header().Set("content-type", "application/json")
			json.NewEncoder(w).Encode(task)
		}
	}
}

func showPokemons(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	firstID, err := strconv.Atoi(vars["id1"])
	if err != nil {
		fmt.Fprintf(w, "Invalid ID 1")
		return
	}
	secondID, err := strconv.Atoi(vars["id2"])
	if err != nil {
		fmt.Fprintf(w, "Invalid ID 2")
		return
	}
	var lista_pokemon []pokemon
	for _, task := range pokemons {
		if task.ID == firstID || task.ID == secondID {
			lista_pokemon = append(lista_pokemon, task)
		}
	}

	w.Header().Set("content-type", "application/json")
	json.NewEncoder(w).Encode(lista_pokemon)

}

func indexRoute(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Welcome to the Pokemon MatchUp API")
}

type botTrain struct {
	xTestData  mat.Matrix `json:XTestData`
	yTestData  mat.Matrix `json:YTestData`
	parameters mat.Matrix `json:Parameters`
	bias       float64    `json:Bias`
}

type Result struct {
	Prediction mat.Matrix `json:Prediction`
	Accuracy   float64    `json:Accuracy`
}

var bot botTrain
var data LogRegression

func trainData(w http.ResponseWriter, r *http.Request) {

	w.Header().Set("content-type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(bot)
}

func predictWinner(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID1, err := strconv.Atoi(vars["id1"])
	taskID2, err := strconv.Atoi(vars["id2"])
	if err != nil {
		fmt.Fprintf(w, "Invalid ID")
		return
	}
	var pokemon1 pokemon
	var pokemon2 pokemon
	for _, poke := range pokemons {
		if poke.ID == taskID1 {
			pokemon1 = poke
		}
		if poke.ID == taskID2 {
			pokemon2 = poke
		}
	}

	var util []float64
	util = append(util, float64(pokemon1.Hp))
	util = append(util, float64(pokemon1.Attack))
	util = append(util, float64(pokemon2.Hp))
	util = append(util, float64(pokemon2.Attack))
	n := mat.NewDense(1, 4, util)

	predictions, accuracyPredict := data.predict(n, nil)

	var result = Result{predictions, accuracyPredict}
	matPrint(predictions)
	w.Header().Set("content-type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(result)

}

func main() {
	wg.Add(1)

	router := mux.NewRouter().StrictSlash(true)

	fmt.Println("Ingrese el puerto")
	fmt.Scanf("%s", &port)
	var ports_list = []string{"4000", "4001", "4002", "4003", "4004"}

	for _, p := range ports_list {
		if p != port {
			remotes = append(remotes, p)
		}
	}

	if port == "4000" {
		/*
			filename := "./Pokemon_matchups.csv"
			split := 18515
			xTrain := make(chan mat.Matrix)
			xTest := make(chan mat.Matrix)
			yTrain := make(chan mat.Matrix)
			yTest := make(chan mat.Matrix)

			//set weights
			weights := make([]float64, 4)
			for i := range weights {
				weights[i] = 1
			}

			weightsData := mat.NewDense(4, 1, weights)

			go trainTestSplit(filename, split, xTrain, xTest, yTrain, yTest)

			xTrainData := <-xTrain
			xTestData := <-xTest
			yTrainData := <-yTrain
			yTestData := <-yTest

			data2 := LogRegression{0.0001, 50000, weightsData, 0.0}

			parameters, bias := data2.fit(xTrainData, yTrainData)

			_, accuracyPredict := data2.predict(xTestData, yTestData)

			fmt.Println("Accuracy predict: ", accuracyPredict, "%")
			bot = botTrain{xTestData, yTestData, parameters, bias}

			data = data2
		*/

		router.HandleFunc("/", indexRoute)
		router.HandleFunc("/pokemons", getPokemons).Methods("GET")
		router.HandleFunc("/pokemons", createPokemon).Methods("POST")
		router.HandleFunc("/pokemons/{id}", getPokemonWithID).Methods("GET")
		router.HandleFunc("/pokemons/{id1}/{id2}", showPokemons).Methods("POST")
		router.HandleFunc("/pokemons/winner/{id1}/{id2}", predictWinner).Methods("GET")
		log.Fatal(http.ListenAndServe(":"+port, router))

	}

	router.HandleFunc("/", indexRoute)
	router.HandleFunc("/pokemons", getPokemons).Methods("GET")
	router.HandleFunc("/pokemons", createPokemon).Methods("POST")
	router.HandleFunc("/pokemons/{id}", getPokemonWithID).Methods("GET")
	router.HandleFunc("/pokemons/{id1}/{id2}", showPokemons).Methods("POST")
	router.HandleFunc("/pokemons/winner/{id1}/{id2}", predictWinner).Methods("GET")
	log.Fatal(http.ListenAndServe(":"+port, router))

}
