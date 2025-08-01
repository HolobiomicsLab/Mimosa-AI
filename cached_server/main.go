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
		log.Printf("[CACHE] WARNING: Failed to create storage directory: %v", err)
	} else {
		log.Printf("[CACHE] Storage directory created/verified: %s", storagePath)
	}
	
	// Load existing cache entries
	log.Printf("[CACHE] Loading existing cache entries...")
	cache.loadIndex()
	
	return cache
}

// loadIndex loads existing cache entries into memory
func (c *RequestCache) loadIndex() {
	startTime := time.Now()
	loadedCount := 0
	errorCount := 0
	
	err := filepath.Walk(c.storagePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Printf("[CACHE] Error walking path %s: %v", path, err)
			errorCount++
			return nil
		}
		
		if !strings.HasSuffix(path, ".json") {
			return nil
		}
		
		data, err := os.ReadFile(path)
		if err != nil {
			log.Printf("[CACHE] Error reading file %s: %v", path, err)
			errorCount++
			return nil
		}
		
		var req CachedRequest
		if err := json.Unmarshal(data, &req); err != nil {
			log.Printf("[CACHE] Error unmarshaling file %s: %v", path, err)
			errorCount++
			return nil
		}
		
		c.index[req.Hash] = &req
		loadedCount++
		return nil
	})
	
	if err != nil {
		log.Printf("[CACHE] Error during index loading: %v", err)
	}
	
	loadDuration := time.Since(startTime)
	log.Printf("[CACHE] Index loading completed: %d entries loaded, %d errors, took %v", 
		loadedCount, errorCount, loadDuration)
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
	hashStr := fmt.Sprintf("%x", hash)
	
	log.Printf("[CACHE] Generated hash %s...", 
		hashStr[:8], normalized["model"], len(req["messages"].([]interface{})))
	
	return hashStr
}

// Get retrieves a cached response
func (c *RequestCache) Get(hash string) *CachedRequest {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if req, exists := c.index[hash]; exists {
		age := time.Since(req.Timestamp)
		log.Printf("[CACHE] ✓ CACHE HIT: Found cached response for hash %s... (provider: %s, age: %v, id: %s)", 
			hash[:8], req.Provider, age.Round(time.Second), req.ID)
		return req
	}
	
	log.Printf("[CACHE] ✗ CACHE MISS: No cached response found for hash %s...", hash[:8])
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
	data, err := json.MarshalIndent(cached, "", "  ")
	if err != nil {
		log.Printf("[CACHE] ERROR: Failed to marshal cache entry: %v", err)
		return
	}
	
	if err := os.WriteFile(filename, data, 0644); err != nil {
		log.Printf("[CACHE] ERROR: Failed to write cache file %s: %v", filename, err)
		return
	}
	
	// Update index
	c.index[hash] = cached
	
	log.Printf("[CACHE] Cache now contains %d entries", len(c.index))
}

// NewRouterServer creates a new router server
func NewRouterServer(configPath string) (*RouterServer, error) {
	log.Printf("[SERVER] Creating new router server with config: %s", configPath)
	
	config, err := loadConfig(configPath)
	if err != nil {
		log.Printf("[SERVER] ERROR: Failed to load config: %v", err)
		return nil, err
	}
	
	log.Printf("[SERVER] Configuration loaded successfully")
	log.Printf("[SERVER] - Port: %d", config.Server.Port)
	log.Printf("[SERVER] - Storage Path: %s", config.Server.StoragePath)
	log.Printf("[SERVER] - Cache Enabled: %v", config.Server.CacheEnabled)
	log.Printf("[SERVER] - Providers configured: %d", len(config.Providers))
	
	cache := NewRequestCache(config.Server.StoragePath)
	
	server := &RouterServer{
		config:    config,
		cache:     cache,
		providers: config.Providers,
	}
	
	log.Printf("[SERVER] Router server created successfully")
	return server, nil
}

// loadConfig loads configuration from file
func loadConfig(path string) (*Config, error) {
	log.Printf("[CONFIG] Loading configuration from: %s", path)
	
	data, err := os.ReadFile(path)
	if err != nil {
		log.Printf("[CONFIG] ERROR: Failed to read config file: %v", err)
		return nil, err
	}
	
	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		log.Printf("[CONFIG] ERROR: Failed to parse YAML config: %v", err)
		return nil, err
	}
	
	log.Printf("[CONFIG] Configuration parsed successfully")
	return &config, nil
}

// selectProvider selects a provider based on configuration
func (rs *RouterServer) selectProvider() Provider {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	
	// Simple round-robin for now - can be enhanced with weighted selection
	if len(rs.providers) > 0 {
		selected := rs.providers[0]
		log.Printf("[PROVIDER] Selected provider: %s (base_url: %s)", selected.Name, selected.BaseURL)
		return selected
	}
	
	log.Printf("[PROVIDER] ERROR: No providers available")
	return Provider{}
}

// forwardRequest forwards request to selected provider
func (rs *RouterServer) forwardRequest(provider Provider, reqBody []byte, originalHeaders map[string]string) (map[string]interface{}, error) {
	startTime := time.Now()
	client := &http.Client{Timeout: 60 * time.Second}
	
	url := provider.BaseURL + "/v1/chat/completions"
	log.Printf("[PROVIDER] Forwarding request to %s at %s", provider.Name, url)
	
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		log.Printf("[PROVIDER] ERROR: Failed to create request: %v", err)
		return nil, err
	}
	
	// Set required headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+provider.APIKey)
	
	// Copy safe headers from original request
	copiedHeaders := 0
	for key, value := range originalHeaders {
		lowerKey := strings.ToLower(key)
		if lowerKey != "authorization" && lowerKey != "host" && 
		   (strings.HasPrefix(lowerKey, "x-") || lowerKey == "user-agent") {
			req.Header.Set(key, value)
			copiedHeaders++
		}
	}
	resp, err := client.Do(req)
	if err != nil {
		duration := time.Since(startTime)
		log.Printf("[PROVIDER] ERROR: Request to %s failed after %v: %v", provider.Name, duration, err)
		return nil, fmt.Errorf("request failed: %v", err)
	}
	defer resp.Body.Close()
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[PROVIDER] ERROR: Failed to read response body: %v", err)
		return nil, fmt.Errorf("failed to read response: %v", err)
	}
	
	log.Printf("[PROVIDER] Response body size: %d bytes", len(body))
	
	// Log response for debugging
	if resp.StatusCode != 200 {
		log.Printf("[PROVIDER] ERROR: Provider %s returned status %d: %s", 
			provider.Name, resp.StatusCode, string(body))
	}
	
	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		log.Printf("[PROVIDER] ERROR: Invalid JSON response from %s: %v", provider.Name, err)
		return nil, fmt.Errorf("invalid JSON response: %v", err)
	}
	
	// Check for API errors
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("provider error (status %d): %s", resp.StatusCode, string(body))
	}
	
	log.Printf("[PROVIDER] ✓ Successfully processed request via %s", provider.Name)
	return result, nil
}

// handleChatCompletions handles chat completion requests
func (rs *RouterServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	requestStart := time.Now()
	requestID := fmt.Sprintf("req_%d", time.Now().UnixNano())
	
	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("[REQUEST] %s ERROR: Failed to read request body: %v", requestID, err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	
	var requestData map[string]interface{}
	if err := json.Unmarshal(body, &requestData); err != nil {
		log.Printf("[REQUEST] %s ERROR: Invalid JSON request: %v", requestID, err)
		http.Error(w, "Invalid JSON request", http.StatusBadRequest)
		return
	}
	
	// Generate hash for caching
	hash := rs.cache.generateHash(requestData)
	if rs.config.Server.CacheEnabled {
		if cached := rs.cache.Get(hash); cached != nil {
			cacheDuration := time.Since(requestStart)
			log.Printf("[REQUEST] %s ✓ CACHE HIT! Returning cached response", 
				requestID, cacheDuration)
			
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-Cache-Status", "HIT")
			w.Header().Set("X-Cache-Provider", cached.Provider)
			w.Header().Set("X-Cache-Age", fmt.Sprintf("%.0f", time.Since(cached.Timestamp).Seconds()))
			
			if err := json.NewEncoder(w).Encode(cached.Response); err != nil {
				log.Printf("[REQUEST] %s ERROR: Failed to encode cached response: %v", requestID, err)
			}
			
			log.Printf("[REQUEST] %s Request completed via cache", requestID)
			return
		}
	}
	
	// Select provider
	provider := rs.selectProvider()
	if provider.Name == "" {
		log.Printf("[REQUEST] %s ERROR: No providers available", requestID)
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
	log.Printf("[REQUEST] %s Forwarding to provider %s...", requestID, provider.Name)
	response, err := rs.forwardRequest(provider, body, headers)
	if err != nil {
		forwardDuration := time.Since(requestStart)
		log.Printf("[REQUEST] %s ERROR: Provider request failed after %v: %v", 
			requestID, forwardDuration, err)
		http.Error(w, fmt.Sprintf("Provider request failed: %v", err), http.StatusBadGateway)
		return
	}
	
	providerDuration := time.Since(requestStart)
	log.Printf("[REQUEST] %s ✓ Provider response received (took %v)", requestID, providerDuration)
	
	// Cache response if enabled
	if rs.config.Server.CacheEnabled {
		log.Printf("[REQUEST] %s Caching response...", requestID)
		rs.cache.Set(hash, requestData, response, provider.Name)
	} else {
		log.Printf("[REQUEST] %s Skipping cache (disabled)", requestID)
	}
	
	// Return response
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache-Status", "MISS")
	w.Header().Set("X-Provider", provider.Name)
	w.Header().Set("X-Response-Time", fmt.Sprintf("%.3f", providerDuration.Seconds()))
	
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("[REQUEST] %s ERROR: Failed to encode response: %v", requestID, err)
	}
	
	totalDuration := time.Since(requestStart)
	log.Printf("[REQUEST] %s completed successfully (total time: %v)", requestID, totalDuration)
}

// healthCheck provides health check endpoint
func (rs *RouterServer) healthCheck(w http.ResponseWriter, r *http.Request) {
	log.Printf("[HEALTH] Health check requested from %s", r.RemoteAddr)
	
	status := map[string]interface{}{
		"status":        "healthy",
		"timestamp":     time.Now().Unix(),
		"providers":     len(rs.providers),
		"cache_enabled": rs.config.Server.CacheEnabled,
		"cache_entries": len(rs.cache.index),
	}
	
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(status); err != nil {
		log.Printf("[HEALTH] ERROR: Failed to encode health status: %v", err)
	}
	
	log.Printf("[HEALTH] Health check completed")
}

// Start starts the router server
func (rs *RouterServer) Start() error {
	log.Printf("[SERVER] Setting up routes...")
	
	r := mux.NewRouter()
	
	// OpenAI-compatible endpoints
	r.HandleFunc("/v1/chat/completions", rs.handleChatCompletions).Methods("POST")
	r.HandleFunc("/health", rs.healthCheck).Methods("GET")
	
	log.Printf("[SERVER] Routes configured")
	
	// CORS middleware
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			
			if r.Method == "OPTIONS" {
				log.Printf("[CORS] Handling OPTIONS request from %s", r.RemoteAddr)
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
	
	log.Printf("[SERVER] ==========================================")
	log.Printf("[SERVER] Starting OpenAI Router Server")
	log.Printf("[SERVER] ==========================================")
	log.Printf("[SERVER] Port: %d", port)
	log.Printf("[SERVER] Storage path: %s", rs.config.Server.StoragePath)
	log.Printf("[SERVER] Cache enabled: %v", rs.config.Server.CacheEnabled)
	log.Printf("[SERVER] Cache entries loaded: %d", len(rs.cache.index))
	log.Printf("[SERVER] Providers configured: %d", len(rs.providers))
	
	for i, provider := range rs.providers {
		log.Printf("[SERVER] Provider %d: %s (%s)", i+1, provider.Name, provider.BaseURL)
	}
	
	log.Printf("[SERVER] ==========================================")
	log.Printf("[SERVER] Server ready at http://localhost:%d", port)
	log.Printf("[SERVER] Health check: http://localhost:%d/health", port)
	log.Printf("[SERVER] Chat completions: http://localhost:%d/v1/chat/completions", port)
	log.Printf("[SERVER] ==========================================")
	
	return http.ListenAndServe(fmt.Sprintf(":%d", port), r)
}

func main() {
	log.Printf("[MAIN] OpenAI Router Server starting...")
	
	configPath := "config.yaml"
	if len(os.Args) > 1 {
		configPath = os.Args[1]
		log.Printf("[MAIN] Using custom config path: %s", configPath)
	} else {
		log.Printf("[MAIN] Using default config path: %s", configPath)
	}
	
	// Create default config if not exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		log.Printf("[MAIN] Config file not found, creating default config...")
		createDefaultConfig(configPath)
	}
	
	server, err := NewRouterServer(configPath)
	if err != nil {
		log.Fatal("[MAIN] FATAL: Failed to create server:", err)
	}
	
	log.Printf("[MAIN] Server initialization completed")
	
	if err := server.Start(); err != nil {
		log.Fatal("[MAIN] FATAL: Server failed:", err)
	}
}

// createDefaultConfig creates a default configuration file
func createDefaultConfig(path string) {
	log.Printf("[CONFIG] Creating default configuration file...")
	
	config := `server:
  port: 8080
  storage_path: "./cache"
  cache_enabled: true

providers:
  - name: "openai"
	base_url: "https://api.deepseek.com"
	api_key: "your-openai-key"
	weight: 1
`
	
	if err := os.WriteFile(path, []byte(config), 0644); err != nil {
		log.Printf("[CONFIG] ERROR: Failed to create default config: %v", err)
	} else {
		log.Printf("[CONFIG] ✓ Created default config file: %s", path)
	}
}