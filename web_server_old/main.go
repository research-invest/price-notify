package main

import (
	"html/template"
	"io"
	"net/http"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

type Template struct {
	templates *template.Template
}

func (t *Template) Render(w io.Writer, name string, data interface{}, c echo.Context) error {
	return t.templates.ExecuteTemplate(w, name, data)
}

func main() {
	e := echo.New()

	// Middleware
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())

	// Статические файлы
	e.Static("/static", "static")

	// Шаблоны
	t := &Template{
		templates: template.Must(template.ParseGlob("templates/*.html")),
	}
	e.Renderer = t

	// Маршруты
	e.GET("/", handleHome)
	e.GET("/trades", handleGetTrades)
	e.POST("/trades", handleAddTrade)
	e.GET("/stats", handleGetStats)

	// Запуск сервера
	e.Logger.Fatal(e.Start(":8080"))
}

func handleHome(c echo.Context) error {
	return c.Render(http.StatusOK, "home.html", map[string]interface{}{
		"title": "Главная",
	})
}

func handleGetTrades(c echo.Context) error {
	// Здесь будет логика получения сделок из базы данных
	trades := []string{"Trade 1", "Trade 2", "Trade 3"} // Пример данных
	return c.Render(http.StatusOK, "trades.html", map[string]interface{}{
		"title":  "Сделки",
		"trades": trades,
	})
}

func handleAddTrade(c echo.Context) error {
	// Здесь будет логика добавления новой сделки
	// Пример:
	// trade := new(Trade)
	// if err := c.Bind(trade); err != nil {
	//     return err
	// }
	// Добавление trade в базу данных
	return c.JSON(http.StatusOK, map[string]string{"message": "Сделка добавлена"})
}

func handleGetStats(c echo.Context) error {
	// Здесь будет логика получения статистики
	stats := map[string]float64{
		"totalProfit": 1000.0,
		"totalTrades": 50,
	}
	return c.Render(http.StatusOK, "stats.html", map[string]interface{}{
		"title": "Статистика",
		"stats": stats,
	})
}
