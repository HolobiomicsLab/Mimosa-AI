from flask import Flask, request, jsonify
from browser import Browser, create_driver
import threading
import time
import sys
import os

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
                driver = create_driver(headless=True)
                browser_instance = Browser(driver)
                return True
            except Exception as e:
                return False
    return True

@app.route('/api/browser/init', methods=['POST'])
def init_browser_route():
    """Initialize or reinitialize the browser"""
    global browser_instance
    with browser_lock:
        if browser_instance is not None:
            try:
                browser_instance.driver.quit()
            except:
                pass
            browser_instance = None
        
        try:
            driver = create_driver(headless=True)
            browser_instance = Browser(driver)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/navigate', methods=['POST'])
def navigate():
    """Navigate to a URL"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"status": "error", "message": "URL is required"}), 400
    
    with browser_lock:
        try:
            success = browser_instance.go_to(url)
            return jsonify({
                "status": "success" if success else "failed",
                "current_url": browser_instance.get_current_url(),
                "title": browser_instance.get_page_title()
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/content', methods=['GET'])
def get_content():
    """Get page content as text"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            content = browser_instance.get_text()
            return jsonify({
                "status": "success",
                "content": content
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/links', methods=['GET'])
def get_links():
    """Get all navigable links on page"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            links = browser_instance.get_navigable()
            return jsonify({
                "status": "success",
                "links": links
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/click', methods=['POST'])
def click():
    """Click an element by XPath"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    data = request.get_json()
    xpath = data.get('xpath')
    if not xpath:
        return jsonify({"status": "error", "message": "XPath is required"}), 400
    
    with browser_lock:
        try:
            success = browser_instance.click_element(xpath)
            return jsonify({
                "status": "success" if success else "failed",
                "current_url": browser_instance.get_current_url()
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/fill_form', methods=['POST'])
def fill_form():
    """Fill form inputs"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    data = request.get_json()
    inputs = data.get('inputs')
    if not inputs or not isinstance(inputs, list):
        return jsonify({"status": "error", "message": "Inputs array is required"}), 400
    
    with browser_lock:
        try:
            success = browser_instance.fill_form(inputs)
            return jsonify({
                "status": "success" if success else "failed",
                "current_url": browser_instance.get_current_url()
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/screenshot', methods=['GET'])
def screenshot():
    """Take and return screenshot"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            filename = f"screenshot_{int(time.time())}.png"
            success = browser_instance.screenshot(filename)
            if not success:
                return jsonify({"status": "error", "message": "Failed to take screenshot"}), 500
            
            # In a real implementation, you would serve the image file or upload it to storage
            # For this prototype, we'll just return the filename
            return jsonify({
                "status": "success",
                "filename": filename
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/info', methods=['GET'])
def get_info():
    """Get current page info"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            return jsonify({
                "status": "success",
                "current_url": browser_instance.get_current_url(),
                "title": browser_instance.get_page_title()
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/link_valid', methods=['POST'])
def is_link_valid():
    """Check if a link is valid for navigation"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"status": "error", "message": "URL is required"}), 400
    
    with browser_lock:
        try:
            valid = browser_instance.is_link_valid(url)
            return jsonify({
                "status": "success",
                "valid": valid
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/browser/form_inputs', methods=['GET'])
def get_form_inputs():
    """Get all form inputs from current page"""
    if not init_browser():
        return jsonify({"status": "error", "message": "Failed to initialize browser"}), 500
    
    with browser_lock:
        try:
            inputs = browser_instance.get_form_inputs()
            return jsonify({
                "status": "success",
                "inputs": inputs
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    screenshots_dir = os.path.join(os.path.dirname(__file__), '.screenshots')
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)
    
    init_browser()
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port 5000.")
    app.run(host='0.0.0.0', port=5000)