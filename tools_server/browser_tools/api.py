from flask import Flask, request, jsonify
from browser import Browser, create_driver
import threading
import time
import sys
import os
import logging

app = Flask(__name__)

# Global browser instance with thread safety
browser_lock = threading.Lock()
browser_instance = None

def init_browser():
    """Initialize the browser instance if not already created"""
    global browser_instance
    with browser_lock:
        if browser_instance is None:
            try:
                print("Initializing browser instance")
                driver = create_driver(headless=True)
                browser_instance = Browser(driver)
                print("Browser instance created successfully")
                return True
            except Exception as e:
                print(f"Failed to initialize browser: {e}")
                return False
    return True

@app.route('/api/browser/init', methods=['POST'])
def init_browser_route():
    """Initialize or reinitialize the browser"""
    global browser_instance
    print("Received request: /api/browser/init")
    with browser_lock:
        if browser_instance is not None:
            try:
                print("Quitting existing browser instance")
                browser_instance.driver.quit()
            except Exception as e:
                print(f"Error quitting browser: {e}")
            browser_instance = None
        
        try:
            print("Creating new browser instance")
            driver = create_driver(headless=True)
            browser_instance = Browser(driver)
            print("Browser instance reinitialized successfully")
            return jsonify({"status": "success"})
        except Exception as e:
            print(f"Failed to reinitialize browser: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/navigate', methods=['POST'])
def navigate():
    """Navigate to a URL"""
    print("Received request: /api/browser/navigate")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    data = request.get_json()
    url = data.get('url')
    print(f"Navigating to URL: {url}")
    if not url:
        print("URL is required but not provided")
        return jsonify({"status": "error", "message": "URL is required"}), 400
    
    with browser_lock:
        try:
            success = browser_instance.go_to(url)
            print(f"Navigation {'succeeded' if success else 'failed'} for URL: {url}")
            return jsonify({
                "status": "success" if success else "failed",
                "current_url": browser_instance.get_current_url(),
                "title": browser_instance.get_page_title()
            })
        except Exception as e:
            print(f"Error navigating to URL {url}: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/content', methods=['GET'])
def get_content():
    """Get page content as text"""
    print("Received request: /api/browser/content")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            content = browser_instance.get_text()
            print("Fetched page content successfully")
            return jsonify({
                "status": "success",
                "content": content
            })
        except Exception as e:
            print(f"Error fetching page content: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/links', methods=['GET'])
def get_links():
    """Get all navigable links on page"""
    print("Received request: /api/browser/links")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            links = browser_instance.get_navigable()
            print(f"Fetched {len(links)} links from page")
            return jsonify({
                "status": "success",
                "links": links
            })
        except Exception as e:
            print(f"Error fetching links: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/click', methods=['POST'])
def click():
    """Click an element by XPath"""
    print("Received request: /api/browser/click")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    data = request.get_json()
    xpath = data.get('xpath')
    print(f"Clicking element with XPath: {xpath}")
    if not xpath:
        print("XPath is required but not provided")
        return jsonify({"status": "error", "message": "XPath is required"}), 400
    
    with browser_lock:
        try:
            success = browser_instance.click_element(xpath)
            print(f"Click {'succeeded' if success else 'failed'} for XPath: {xpath}")
            return jsonify({
                "status": "success" if success else "failed",
                "current_url": browser_instance.get_current_url()
            })
        except Exception as e:
            print(f"Error clicking element {xpath}: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/fill_form', methods=['POST'])
def fill_form():
    """Fill form inputs"""
    print("Received request: /api/browser/fill_form")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    data = request.get_json()
    inputs = data.get('inputs')
    print(f"Filling form with inputs: {inputs}")
    if not inputs or not isinstance(inputs, list):
        print("Inputs array is required but not provided or invalid")
        return jsonify({"status": "error", "message": "Inputs array is required"}), 400
    
    with browser_lock:
        try:
            success = browser_instance.fill_form(inputs)
            print(f"Form fill {'succeeded' if success else 'failed'}")
            return jsonify({
                "status": "success" if success else "failed",
                "current_url": browser_instance.get_current_url()
            })
        except Exception as e:
            print(f"Error filling form: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/screenshot', methods=['GET'])
def screenshot():
    """Take and return screenshot"""
    print("Received request: /api/browser/screenshot")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            filename = f"screenshot_{int(time.time())}.png"
            print(f"Taking screenshot: {filename}")
            success = browser_instance.screenshot(filename)
            if not success:
                print("Failed to take screenshot")
                return jsonify({"status": "error", "message": "Failed to take screenshot"}), 500

            print(f"Screenshot saved as {filename}")
            return jsonify({
                "status": "success",
                "filename": filename
            })
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/info', methods=['GET'])
def get_info():
    """Get current page info"""
    print("Received request: /api/browser/info")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            print("Fetching current page info")
            return jsonify({
                "status": "success",
                "current_url": browser_instance.get_current_url(),
                "title": browser_instance.get_page_title()
            })
        except Exception as e:
            print(f"Error fetching page info: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/link_valid', methods=['POST'])
def is_link_valid():
    """Check if a link is valid for navigation"""
    print("Received request: /api/browser/link_valid")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    data = request.get_json()
    url = data.get('url')
    print(f"Checking if link is valid: {url}")
    if not url:
        print("URL is required but not provided")
        return jsonify({"status": "error", "message": "URL is required"}), 400
    
    with browser_lock:
        try:
            valid = browser_instance.is_link_valid(url)
            print(f"Link valid: {valid} for URL: {url}")
            return jsonify({
                "status": "success",
                "valid": valid
            })
        except Exception as e:
            print(f"Error checking link validity: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/form_inputs', methods=['GET'])
def get_form_inputs():
    """Get all form inputs from current page"""
    print("Received request: /api/browser/form_inputs")
    if not init_browser():
        print("Failed to initialize browser")
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            inputs = browser_instance.get_form_inputs()
            print(f"Fetched {len(inputs)} form inputs from page")
            return jsonify({
                "status": "success",
                "inputs": inputs
            })
        except Exception as e:
            print(f"Error fetching form inputs: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    screenshots_dir = os.path.join(os.path.dirname(__file__), '.screenshots')
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    print("Starting browser tools API server")
    init_browser()
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port 5000.")
    app.run(host='0.0.0.0', port=port)