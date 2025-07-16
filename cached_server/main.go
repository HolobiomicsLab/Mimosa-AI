package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"gopkg.in/yaml.v2"
)

// Config represents the server configuration
type Config struct {
	Server struct {
		Port         int    `yaml:"port"`
		StoragePath  string `yaml:"storage_path"`
		CacheEnabled bool   `yaml:"cache_enabled"`
	} `yaml:"server"`
	
	Providers []Provider `yaml:"providers"`
}

// Provider represents an OpenAI-compatible API provider
type Provider struct {
	Name    string `yaml:"name"`
	BaseURL string `yaml:"base_url"`
	APIKey  string `yaml:"api_key"`
	Weight  int    `yaml:"weight"`
}

// Request represents a cached request
type CachedRequest struct {
	ID        string                 `json:"id"`
	Hash      string                 `json:"hash"`
	Request   map[string]interface{} `json:"request"`
	Response  map[string]interface{} `json:"response"`
	Provider  string                 `json:"provider"`
	Timestamp time.Time              `json:"timestamp"`
}

// RouterServer manages the routing and caching logic
type RouterServer struct {
	config    *Config
	cache     *RequestCache
	providers []Provider
	mu        sync.RWMutex
}

// RequestCache handles caching operations
type RequestCache struct {
	storagePath string
	mu          sync.RWMutex
	index       map[string]*CachedRequest
}

// NewRequestCache creates a new cache instance
func NewRequestCache(storagePath string) *RequestCache {
	cache := &RequestCache{
		storagePath: storagePath,
		index:       make(map[string]*CachedRequest),
	}
	
	// Create storage directory
	os.MkdirAll(storagePath, 0755)
	
	// Load existing cache entries
	cache.loadIndex()
	
	return cache
}

// loadIndex loads existing cache entries into memory
func (c *RequestCache) loadIndex() {
	filepath.Walk(c.storagePath, func(path string, info os.FileInfo, err error) error {
		if err != nil || !strings.HasSuffix(path, ".json") {
			return nil
		}
		
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		
		var req CachedRequest
		if json.Unmarshal(data, &req) == nil {
			c.index[req.Hash] = &req
		}
		
		return nil
	})
}

// generateHash creates a hash for request similarity matching
func (c *RequestCache) generateHash(req map[string]interface{}) string {
	// Normalize request for consistent hashing
	normalized := make(map[string]interface{})
	
	// Include key fields that affect response
	if model, ok := req["model"]; ok {
		normalized["model"] = model
	}
	if messages, ok := req["messages"]; ok {
		normalized["messages"] = messages
	}
	if temperature, ok := req["temperature"]; ok {
		normalized["temperature"] = temperature
	}
	if maxTokens, ok := req["max_tokens"]; ok {
		normalized["max_tokens"] = maxTokens
	}
	
	data, _ := json.Marshal(normalized)
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash)
}

// Get retrieves a cached response
func (c *RequestCache) Get(hash string) *CachedRequest {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if req, exists := c.index[hash]; exists {
		return req
	}
	return nil
}

// Set stores a request/response pair in cache
func (c *RequestCache) Set(hash string, req map[string]interface{}, resp map[string]interface{}, provider string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	cached := &CachedRequest{
		ID:        fmt.Sprintf("%d", time.Now().UnixNano()),
		Hash:      hash,
		Request:   req,
		Response:  resp,
		Provider:  provider,
		Timestamp: time.Now(),
	}
	
	// Store to disk
	filename := filepath.Join(c.storagePath, fmt.Sprintf("%s.json", cached.ID))
	data, _ := json.MarshalIndent(cached, "", "  ")
	os.WriteFile(filename, data, 0644)
	
	// Update index
	c.index[hash] = cached
}

// NewRouterServer creates a new router server
func NewRouterServer(configPath string) (*RouterServer, error) {
	config, err := loadConfig(configPath)
	if err != nil {
		return nil, err
	}
	
	cache := NewRequestCache(config.Server.StoragePath)
	
	return &RouterServer{
		config:    config,
		cache:     cache,
		providers: config.Providers,
	}, nil
}

// loadConfig loads configuration from file
func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	
	var config Config
	err = yaml.Unmarshal(data, &config)
	return &config, err
}

// selectProvider selects a provider based on configuration
func (rs *RouterServer) selectProvider() Provider {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	
	// Simple round-robin for now - can be enhanced with weighted selection
	if len(rs.providers) > 0 {
		return rs.providers[0]
	}
	
	return Provider{}
}

// forwardRequest forwards request to selected provider
func (rs *RouterServer) forwardRequest(provider Provider, reqBody []byte, headers map[string]string) (map[string]interface{}, error) {
	client := &http.Client{Timeout: 60 * time.Second}
	
	url := provider.BaseURL + "/v1/chat/completions"
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}
	
	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+provider.APIKey)
	
	// Copy relevant headers from original request
	for key, value := range headers {
		if strings.HasPrefix(strings.ToLower(key), "x-") {
			req.Header.Set(key, value)
		}
	}
	
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	
	var result map[string]interface{}
	err = json.Unmarshal(body, &result)
	return result, err
}

// handleChatCompletions handles chat completion requests
func (rs *RouterServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	
	// Parse request
	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		http.Error(w, "Invalid JSON request", http.StatusBadRequest)
		return
	}
	
	// Generate hash for caching
	hash := rs.cache.generateHash(requestData)
	
	// Check cache if enabled
	if rs.config.Server.CacheEnabled {
		if cached := rs.cache.Get(hash); cached != nil {
			log.Printf("Cache hit for request hash: %s", hash)
			
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Cache-Status", "HIT")
			json.NewEncoder(w).Encode(cached.Response)
			return
		}
	}
	
	// Select provider
	provider := rs.selectProvider()
	if provider.Name == "" {
		http.Error(w, "No providers available", http.StatusServiceUnavailable)
		return
	}
	
	// Extract headers
	headers := make(map[string]string)
	for key, values := range r.Header {
		if len(values) > 0 {
			headers[key] = values[0]
		}
	}
	
	// Forward request
	response, err := rs.forwardRequest(provider, body, headers)
	if err != nil {
		log.Printf("Error forwarding request: %v", err)
		http.Error(w, "Provider request failed", http.StatusBadGateway)
		return
	}
	
	// Cache response if enabled
	if rs.config.Server.CacheEnabled {
		rs.cache.Set(hash, requestData, response, provider.Name)
	}
	
	// Return response
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache-Status", "MISS")
	w.Header().Set("X-Provider", provider.Name)
	json.NewEncoder(w).Encode(response)
}

// healthCheck provides health check endpoint
func (rs *RouterServer) healthCheck(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"providers": len(rs.providers),
		"cache_enabled": rs.config.Server.CacheEnabled,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// Start starts the router server
func (rs *RouterServer) Start() error {
	r := mux.NewRouter()
	
	// OpenAI-compatible endpoints
	r.HandleFunc("/v1/chat/completions", rs.handleChatCompletions).Methods("POST")
	r.HandleFunc("/health", rs.healthCheck).Methods("GET")
	
	// CORS middleware
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	})
	
	port := rs.config.Server.Port
	if port == 0 {
		port = 8080
	}
	
	log.Printf("Starting OpenAI Router on port %d", port)
	log.Printf("Storage path: %s", rs.config.Server.StoragePath)
	log.Printf("Cache enabled: %v", rs.config.Server.CacheEnabled)
	log.Printf("Providers: %d", len(rs.providers))
	
	return http.ListenAndServe(fmt.Sprintf(":%d", port), r)
}

func main() {
	configPath := "config.yaml"
	if len(os.Args) > 1 {
		configPath = os.Args[1]
	}
	
	// Create default config if not exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		createDefaultConfig(configPath)
	}
	
	server, err := NewRouterServer(configPath)
	if err != nil {
		log.Fatal("Failed to create server:", err)
	}
	
	if err := server.Start(); err != nil {
		log.Fatal("Server failed:", err)
	}
}

// createDefaultConfig creates a default configuration file
func createDefaultConfig(path string) {
	config := `server:
  port: 8080
  storage_path: "./cache"
  cache_enabled: true

providers:
  - name: "deepseek"
    base_url: "https://api.deepseek.com"
    api_key: "api-key-here"
    weight: 1
`
	
	os.WriteFile(path, []byte(config), 0644)
	log.Printf("Created default config file: %s", path)
}