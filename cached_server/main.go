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
	log.Printf("[CACHE] Initializing cache with storage path: %s", storagePath)
	
	cache := &RequestCache{
		storagePath: storagePath,
		index:       make(map[string]*CachedRequest),
	}
	
	// Create storage directory
	if err := os.MkdirAll(storagePath, 0755); err != nil {
		log.Printf("[CACHE] Error creating storage directory: %v", err)
	} else {
		log.Printf("[CACHE] Storage directory created/verified: %s", storagePath)
	}
	
	// Load existing cache entries
	cache.loadIndex()
	
	return cache
}

// loadIndex loads existing cache entries into memory
func (c *RequestCache) loadIndex() {
	log.Printf("[CACHE] Loading existing cache entries from: %s", c.storagePath)
	
	loadedCount := 0
	err := filepath.Walk(c.storagePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Printf("[CACHE] Error walking path %s: %v", path, err)
			return nil
		}
		
		if !strings.HasSuffix(path, ".json") {
			return nil
		}
		
		log.Printf("[CACHE] Loading cache file: %s", path)
		
		data, err := os.ReadFile(path)
		if err != nil {
			log.Printf("[CACHE] Error reading cache file %s: %v", path, err)
			return nil
		}
		
		var req CachedRequest
		if err := json.Unmarshal(data, &req); err != nil {
			log.Printf("[CACHE] Error unmarshaling cache file %s: %v", path, err)
			return nil
		}
		
		c.index[req.Hash] = &req
		loadedCount++
		log.Printf("[CACHE] Loaded cache entry: ID=%s, Hash=%s, Provider=%s, Timestamp=%s", 
			req.ID, req.Hash[:16]+"...", req.Provider, req.Timestamp.Format(time.RFC3339))
		
		return nil
	})
	
	if err != nil {
		log.Printf("[CACHE] Error during cache loading: %v", err)
	}
	
	log.Printf("[CACHE] Cache index loaded successfully. Total entries: %d", loadedCount)
}

// generateHash creates a hash for request similarity matching
func (c *RequestCache) generateHash(req map[string]interface{}) string {
	log.Printf("[CACHE] Generating hash for request")
	
	// Normalize request for consistent hashing
	normalized := make(map[string]interface{})
	
	// Include key fields that affect response
	if model, ok := req["model"]; ok {
		normalized["model"] = model
		log.Printf("[CACHE] Hash includes model: %v", model)
	}
	if messages, ok := req["messages"]; ok {
		normalized["messages"] = messages
		if msgSlice, ok := messages.([]interface{}); ok {
			log.Printf("[CACHE] Hash includes %d messages", len(msgSlice))
		}
	}
	if temperature, ok := req["temperature"]; ok {
		normalized["temperature"] = temperature
		log.Printf("[CACHE] Hash includes temperature: %v", temperature)
	}
	if maxTokens, ok := req["max_tokens"]; ok {
		normalized["max_tokens"] = maxTokens
		log.Printf("[CACHE] Hash includes max_tokens: %v", maxTokens)
	}
	
	data, _ := json.Marshal(normalized)
	hash := sha256.Sum256(data)
	hashStr := fmt.Sprintf("%x", hash)
	
	log.Printf("[CACHE] Generated hash: %s (from %d bytes of normalized data)", hashStr[:16]+"...", len(data))
	
	return hashStr
}

// Get retrieves a cached response
func (c *RequestCache) Get(hash string) *CachedRequest {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	log.Printf("[CACHE] Checking cache for hash: %s", hash[:16]+"...")
	
	if req, exists := c.index[hash]; exists {
		age := time.Since(req.Timestamp)
		log.Printf("[CACHE] Cache HIT! Found entry: ID=%s, Provider=%s, Age=%s", 
			req.ID, req.Provider, age.String())
		
		// Log some details about the cached request
		if model, ok := req.Request["model"]; ok {
			log.Printf("[CACHE] Cached request model: %v", model)
		}
		if messages, ok := req.Request["messages"]; ok {
			if msgSlice, ok := messages.([]interface{}); ok {
				log.Printf("[CACHE] Cached request has %d messages", len(msgSlice))
			}
		}
		
		return req
	}
	
	log.Printf("[CACHE] Cache MISS for hash: %s", hash[:16]+"...")
	return nil
}

// Set stores a request/response pair in cache
func (c *RequestCache) Set(hash string, req map[string]interface{}, resp map[string]interface{}, provider string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	log.Printf("[CACHE] Storing new cache entry for hash: %s", hash[:16]+"...")
	
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
	data, err := json.MarshalIndent(cached, "", "  ")
	if err != nil {
		log.Printf("[CACHE] Error marshaling cache entry: %v", err)
		return
	}
	
	if err := os.WriteFile(filename, data, 0644); err != nil {
		log.Printf("[CACHE] Error writing cache file %s: %v", filename, err)
		return
	}
	
	// Update index
	c.index[hash] = cached
	
	log.Printf("[CACHE] Cache entry saved successfully: ID=%s, File=%s, Provider=%s", 
		cached.ID, filename, provider)
	log.Printf("[CACHE] Total cache entries: %d", len(c.index))
}

// NewRouterServer creates a new router server
func NewRouterServer(configPath string) (*RouterServer, error) {
	log.Printf("[SERVER] Loading configuration from: %s", configPath)
	
	config, err := loadConfig(configPath)
	if err != nil {
		log.Printf("[SERVER] Error loading config: %v", err)
		return nil, err
	}
	
	log.Printf("[SERVER] Configuration loaded successfully")
	log.Printf("[SERVER] Port: %d", config.Server.Port)
	log.Printf("[SERVER] Storage Path: %s", config.Server.StoragePath)
	log.Printf("[SERVER] Cache Enabled: %v", config.Server.CacheEnabled)
	log.Printf("[SERVER] Providers: %d", len(config.Providers))
	
	for i, provider := range config.Providers {
		log.Printf("[SERVER] Provider %d: %s (%s) - Weight: %d", 
			i+1, provider.Name, provider.BaseURL, provider.Weight)
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
	
	log.Printf("[PROVIDER] Selecting provider from %d available providers", len(rs.providers))
	
	// Simple round-robin for now - can be enhanced with weighted selection
	if len(rs.providers) > 0 {
		selected := rs.providers[0]
		log.Printf("[PROVIDER] Selected provider: %s (%s)", selected.Name, selected.BaseURL)
		return selected
	}
	
	log.Printf("[PROVIDER] No providers available!")
	return Provider{}
}

// forwardRequest forwards request to selected provider
func (rs *RouterServer) forwardRequest(provider Provider, reqBody []byte, headers map[string]string) (map[string]interface{}, error) {
	log.Printf("[REQUEST] Forwarding request to provider: %s", provider.Name)
	log.Printf("[REQUEST] Provider URL: %s", provider.BaseURL)
	log.Printf("[REQUEST] Request size: %d bytes", len(reqBody))
	
	client := &http.Client{Timeout: 60 * time.Second}
	
	url := provider.BaseURL + "/v1/chat/completions"
	log.Printf("[REQUEST] Full endpoint URL: %s", url)
	
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		log.Printf("[REQUEST] Error creating HTTP request: %v", err)
		return nil, err
	}
	
	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+provider.APIKey[:min(len(provider.APIKey), 20)]+"...")
	log.Printf("[REQUEST] Set Authorization header (key prefix: %s...)", provider.APIKey[:min(len(provider.APIKey), 8)])
	
	// Copy relevant headers from original request
	copiedHeaders := 0
	for key, value := range headers {
		if strings.HasPrefix(strings.ToLower(key), "x-") {
			req.Header.Set(key, value)
			copiedHeaders++
		}
	}
	log.Printf("[REQUEST] Copied %d custom headers", copiedHeaders)
	
	start := time.Now()
	resp, err := client.Do(req)
	duration := time.Since(start)
	
	if err != nil {
		log.Printf("[REQUEST] HTTP request failed after %s: %v", duration.String(), err)
		return nil, err
	}
	defer resp.Body.Close()
	
	log.Printf("[REQUEST] HTTP request completed in %s, status: %s", duration.String(), resp.Status)
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[REQUEST] Error reading response body: %v", err)
		return nil, err
	}
	
	log.Printf("[REQUEST] Response size: %d bytes", len(body))
	
	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		log.Printf("[REQUEST] Error unmarshaling response: %v", err)
		return nil, err
	}
	
	// Log some response details
	if usage, ok := result["usage"]; ok {
		log.Printf("[REQUEST] Response usage: %v", usage)
	}
	if choices, ok := result["choices"]; ok {
		if choiceSlice, ok := choices.([]interface{}); ok {
			log.Printf("[REQUEST] Response contains %d choices", len(choiceSlice))
		}
	}
	
	log.Printf("[REQUEST] Successfully parsed response from provider: %s", provider.Name)
	return result, err
}

// handleChatCompletions handles chat completion requests
func (rs *RouterServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	requestStart := time.Now()
	clientIP := r.RemoteAddr
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		clientIP = forwarded
	}
	
	log.Printf("[HANDLER] New chat completion request from %s", clientIP)
	log.Printf("[HANDLER] Request method: %s, URL: %s", r.Method, r.URL.String())
	log.Printf("[HANDLER] User-Agent: %s", r.Header.Get("User-Agent"))
	
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("[HANDLER] Error reading request body: %v", err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	
	log.Printf("[HANDLER] Request body size: %d bytes", len(body))
	
	// Parse request
	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		log.Printf("[HANDLER] Error parsing JSON request: %v", err)
		http.Error(w, "Invalid JSON request", http.StatusBadRequest)
		return
	}
	
	// Log request details
	if model, ok := requestData["model"]; ok {
		log.Printf("[HANDLER] Request model: %v", model)
	}
	if messages, ok := requestData["messages"]; ok {
		if msgSlice, ok := messages.([]interface{}); ok {
			log.Printf("[HANDLER] Request contains %d messages", len(msgSlice))
			for i, msg := range msgSlice {
				if msgMap, ok := msg.(map[string]interface{}); ok {
					if role, ok := msgMap["role"]; ok {
						if content, ok := msgMap["content"]; ok {
							if contentStr, ok := content.(string); ok {
								log.Printf("[HANDLER] Message %d: role=%v, content_length=%d", i+1, role, len(contentStr))
							}
						}
					}
				}
			}
		}
	}
	if temperature, ok := requestData["temperature"]; ok {
		log.Printf("[HANDLER] Request temperature: %v", temperature)
	}
	if maxTokens, ok := requestData["max_tokens"]; ok {
		log.Printf("[HANDLER] Request max_tokens: %v", maxTokens)
	}
	
	// Generate hash for caching
	hash := rs.cache.generateHash(requestData)
	log.Printf("[HANDLER] Generated request hash: %s", hash[:16]+"...")
	
	// Check cache if enabled
	if rs.config.Server.CacheEnabled {
		log.Printf("[HANDLER] Cache is enabled, checking for existing response")
		
		if cached := rs.cache.Get(hash); cached != nil {
			cacheAge := time.Since(cached.Timestamp)
			totalDuration := time.Since(requestStart)
			
			log.Printf("[HANDLER] ✅ CACHE HIT! Serving cached response")
			log.Printf("[HANDLER] Cache entry age: %s", cacheAge.String())
			log.Printf("[HANDLER] Original provider: %s", cached.Provider)
			log.Printf("[HANDLER] Request completed in %s (cache hit)", totalDuration.String())
			
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Cache-Status", "HIT")
			w.Header().Set("X-Cache-Age", cacheAge.String())
			w.Header().Set("X-Original-Provider", cached.Provider)
			json.NewEncoder(w).Encode(cached.Response)
			return
		} else {
			log.Printf("[HANDLER] ❌ CACHE MISS - proceeding to provider")
		}
	} else {
		log.Printf("[HANDLER] Cache is disabled, proceeding directly to provider")
	}
	
	// Select provider
	provider := rs.selectProvider()
	if provider.Name == "" {
		log.Printf("[HANDLER] ERROR: No providers available")
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
	log.Printf("[HANDLER] Extracted %d headers from request", len(headers))
	
	// Forward request
	log.Printf("[HANDLER] Forwarding request to provider: %s", provider.Name)
	response, err := rs.forwardRequest(provider, body, headers)
	if err != nil {
		log.Printf("[HANDLER] ERROR: Provider request failed: %v", err)
		http.Error(w, "Provider request failed", http.StatusBadGateway)
		return
	}
	
	log.Printf("[HANDLER] ✅ Received response from provider: %s", provider.Name)
	
	// Cache response if enabled
	if rs.config.Server.CacheEnabled {
		log.Printf("[HANDLER] Caching response for future requests")
		rs.cache.Set(hash, requestData, response, provider.Name)
		log.Printf("[HANDLER] Response cached successfully")
	} else {
		log.Printf("[HANDLER] Caching disabled, not storing response")
	}
	
	totalDuration := time.Since(requestStart)
	log.Printf("[HANDLER] Request completed in %s (cache miss + provider call)", totalDuration.String())
	
	// Return response
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache-Status", "MISS")
	w.Header().Set("X-Provider", provider.Name)
	w.Header().Set("X-Response-Time", totalDuration.String())
	json.NewEncoder(w).Encode(response)
	
	log.Printf("[HANDLER] Response sent to client %s", clientIP)
}

// healthCheck provides health check endpoint
func (rs *RouterServer) healthCheck(w http.ResponseWriter, r *http.Request) {
	log.Printf("[HEALTH] Health check requested from %s", r.RemoteAddr)
	
	cacheEntries := 0
	if rs.cache != nil {
		rs.cache.mu.RLock()
		cacheEntries = len(rs.cache.index)
		rs.cache.mu.RUnlock()
	}
	
	status := map[string]interface{}{
		"status":        "healthy",
		"timestamp":     time.Now().Unix(),
		"providers":     len(rs.providers),
		"cache_enabled": rs.config.Server.CacheEnabled,
		"cache_entries": cacheEntries,
	}
	
	log.Printf("[HEALTH] Health status: providers=%d, cache_enabled=%v, cache_entries=%d", 
		len(rs.providers), rs.config.Server.CacheEnabled, cacheEntries)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// Start starts the router server
func (rs *RouterServer) Start() error {
	log.Printf("[SERVER] Initializing HTTP router")
	
	r := mux.NewRouter()
	
	// OpenAI-compatible endpoints
	r.HandleFunc("/v1/chat/completions", rs.handleChatCompletions).Methods("POST")
	r.HandleFunc("/health", rs.healthCheck).Methods("GET")
	
	log.Printf("[SERVER] Registered endpoints:")
	log.Printf("[SERVER]   POST /v1/chat/completions - Chat completions")
	log.Printf("[SERVER]   GET  /health - Health check")
	
	// CORS middleware
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			log.Printf("[CORS] %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)
			
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			
			if r.Method == "OPTIONS" {
				log.Printf("[CORS] Responding to preflight request")
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
	
	log.Printf("[SERVER] ========================================")
	log.Printf("[SERVER] 🚀 Starting OpenAI Router Server")
	log.Printf("[SERVER] ========================================")
	log.Printf("[SERVER] Port: %d", port)
	log.Printf("[SERVER] Storage path: %s", rs.config.Server.StoragePath)
	log.Printf("[SERVER] Cache enabled: %v", rs.config.Server.CacheEnabled)
	log.Printf("[SERVER] Providers: %d", len(rs.providers))
	log.Printf("[SERVER] ========================================")
	
	return http.ListenAndServe(fmt.Sprintf(":%d", port), r)
}

func main() {
	log.Printf("[MAIN] Starting application...")
	
	configPath := "config.yaml"
	if len(os.Args) > 1 {
		configPath = os.Args[1]
		log.Printf("[MAIN] Using config file from argument: %s", configPath)
	} else {
		log.Printf("[MAIN] Using default config file: %s", configPath)
	}
	
	// Create default config if not exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		log.Printf("[MAIN] Config file not found, creating default config")
		createDefaultConfig(configPath)
	} else {
		log.Printf("[MAIN] Config file exists: %s", configPath)
	}
	
	server, err := NewRouterServer(configPath)
	if err != nil {
		log.Fatal("[MAIN] Failed to create server:", err)
	}
	
	log.Printf("[MAIN] Server created successfully, starting...")
	
	if err := server.Start(); err != nil {
		log.Fatal("[MAIN] Server failed:", err)
	}
}

// createDefaultConfig creates a default configuration file
func createDefaultConfig(path string) {
	log.Printf("[CONFIG] Creating default configuration file: %s", path)
	
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
	
	if err := os.WriteFile(path, []byte(config), 0644); err != nil {
		log.Printf("[CONFIG] Error creating default config: %v", err)
	} else {
		log.Printf("[CONFIG] Created default config file: %s", path)
	}
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}