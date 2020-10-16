package main

import (
	"fmt"
	"log"
	"net/http"
	"web_api/handlers"

	"github.com/go-chi/chi"
	"github.com/go-chi/chi/middleware"
)

func main() {
	fmt.Println("Starting server on port :3000")
	router := chi.NewRouter()
	router.Use(middleware.Logger)

	// Set up static file serving
	fs := http.FileServer(http.Dir("./html"))
	router.Handle("/*", fs)
	router.Get("/chatbot", handlers.GetResponse)

	//run it on port 8080
	err := http.ListenAndServe(":3000", router)
	if err != nil {
		log.Fatal(err)
	}
}
