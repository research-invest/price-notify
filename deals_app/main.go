package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gin-gonic/gin"
	_ "github.com/go-sql-driver/mysql"
)

var templates map[string]*template.Template
var db *sql.DB

type Trade struct {
	ID        int64
	Symbol    string
	Price     float64
	Amount    float64
	TradeType string // "long" или "short"
}

type Config struct {
	DB struct {
		Host     string `json:"host"`
		Database string `json:"database"`
		User     string `json:"user"`
		Password string `json:"password"`
	} `json:"db"`
}

func loadConfig() (Config, error) {
	var config Config
	configPath := filepath.Join("..", "config.json")
	file, err := os.Open(configPath)
	if err != nil {
		return config, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	err = decoder.Decode(&config)
	return config, err
}

func loadTemplates() error {
	templates = make(map[string]*template.Template)

	layoutFiles, err := filepath.Glob("templates/layout.html")
	if err != nil {
		return err
	}

	includeFiles, err := filepath.Glob("templates/*.html")
	if err != nil {
		return err
	}

	for _, file := range includeFiles {
		fileName := filepath.Base(file)
		if fileName == "layout.html" {
			continue
		}
		templateName := strings.TrimSuffix(fileName, ".html")
		files := append(layoutFiles, file)
		templates[templateName] = template.Must(template.ParseFiles(files...))
	}

	return nil
}

func initDB(config Config) error {
	var err error
	dsn := fmt.Sprintf("%s:%s@tcp(%s)/%s", config.DB.User, config.DB.Password, config.DB.Host, config.DB.Database)
	db, err = sql.Open("mysql", dsn)
	if err != nil {
		return err
	}
	return nil
}

func main() {
	if err := loadTemplates(); err != nil {
		panic(err)
	}

	config, err := loadConfig()
	if err != nil {
		log.Fatal(err)
	}

	if err := initDB(config); err != nil {
		panic(err)
	}
	defer db.Close()

	r := gin.Default()

	// Статические файлы
	r.Static("/static", "./static")

	// Маршруты
	r.GET("/", handleHome)
	r.GET("/trades", handleGetTrades)
	r.POST("/trades", handleAddTrade)
	r.GET("/stats", handleStats)

	// Запуск сервера
	r.Run(":8080")
}

func render(c *gin.Context, templateName string, data gin.H) {
	tmpl := templates[templateName]
	if tmpl == nil {
		c.String(http.StatusInternalServerError, "Template not found")
		return
	}
	c.Status(http.StatusOK)
	if err := tmpl.Execute(c.Writer, data); err != nil {
		c.String(http.StatusInternalServerError, err.Error())
	}
}

func handleHome(c *gin.Context) {
	render(c, "home", gin.H{
		"title": "Главная",
	})
}

func handleGetTrades(c *gin.Context) {
	trades, err := getAllTrades()
	if err != nil {
		c.String(http.StatusInternalServerError, err.Error())
		return
	}

	render(c, "trades", gin.H{
		"title":  "Сделки",
		"trades": trades,
	})
}

func handleAddTrade(c *gin.Context) {
	var trade Trade
	if err := c.ShouldBind(&trade); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	id, err := addTrade(trade)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"id": id})
}

func handleStats(c *gin.Context) {
	stats := map[string]float64{
		"totalProfit": 1000.0,
		"totalTrades": 50,
	}
	render(c, "stats", gin.H{
		"title": "Статистика",
		"stats": stats,
	})
}

func getAllTrades() ([]Trade, error) {
	rows, err := db.Query("SELECT id, symbol, price, amount, trade_type FROM trades")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var trades []Trade
	for rows.Next() {
		var t Trade
		if err := rows.Scan(&t.ID, &t.Symbol, &t.Price, &t.Amount, &t.TradeType); err != nil {
			return nil, err
		}
		trades = append(trades, t)
	}

	return trades, nil
}

func addTrade(trade Trade) (int64, error) {
	result, err := db.Exec("INSERT INTO trades (symbol, price, amount, trade_type) VALUES (?, ?, ?, ?)",
		trade.Symbol, trade.Price, trade.Amount, trade.TradeType)
	if err != nil {
		return 0, err
	}

	return result.LastInsertId()
}
