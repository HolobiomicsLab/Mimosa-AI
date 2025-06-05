import requests
import time

API_BASE_URL = "http://localhost:5000"

def test_api_endpoints():
    """Test all browser API endpoints"""
    print("Starting browser API tests...\n")
    
    # Test initialization
    print("Testing /api/browser/init...")
    try:
        response = requests.post(f"{API_BASE_URL}/api/browser/init")
        assert response.status_code == 200
        assert response.json().get("status") == "success"
        print("✓ Successfully initialized browser\n")
    except Exception as e:
        print(f"✗ Initialization failed: {str(e)}\n")

    # Test navigation
    print("Testing /api/browser/navigate with example.com...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/browser/navigate",
            json={"url": "https://example.com"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ["success", "failed"]
        print(f"✓ Navigation test completed (status: {data.get('status')})\n")
    except Exception as e:
        print(f"✗ Navigation test failed: {str(e)}\n")

    # Test content
    print("Testing /api/browser/content...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/browser/content")
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        print("✓ Content test completed\n")
    except Exception as e:
        print(f"✗ Content test failed: {str(e)}\n")

    # Test links
    print("Testing /api/browser/links...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/browser/links")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data.get("links", []), list)
        print("✓ Links test completed\n")
    except Exception as e:
        print(f"✗ Links test failed: {str(e)}\n")

    # Test link validation
    print("Testing /api/browser/link_valid...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/browser/link_valid",
            json={"url": "https://example.com"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        print(f"✓ Link validation test completed (valid: {data.get('valid')})\n")
    except Exception as e:
        print(f"✗ Link validation test failed: {str(e)}\n")

    # Test form inputs
    print("Testing /api/browser/form_inputs...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/browser/form_inputs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data.get("inputs", []), list)
        print("✓ Form inputs test completed\n")
    except Exception as e:
        print(f"✗ Form inputs test failed: {str(e)}\n")

    # Test screenshot
    print("Testing /api/browser/screenshot...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/browser/screenshot")
        assert response.status_code == 200
        data = response.json()
        assert "filename" in data
        print(f"✓ Screenshot test completed (filename: {data.get('filename')})\n")
    except Exception as e:
        print(f"✗ Screenshot test failed: {str(e)}\n")

    print("All tests completed!")

if __name__ == "__main__":
    # Wait briefly to ensure server is up if just started
    time.sleep(1)
    test_api_endpoints()
