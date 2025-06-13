#!/usr/bin/env python3
"""
Comprehensive test client for CSV API
Tests all routes with edge cases that will break in production
"""

import requests
import json
import io
import sys
from typing import Dict, Any

class CSVAPITester:
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test results"""
        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {details}")
        self.test_results.append({"test": test_name, "success": success, "details": details})
    
    def test_create_csv(self):
        """Test CSV creation with various scenarios"""
        print("\n=== Testing CSV Creation ===")
        
        # Valid creation
        payload = {
            "name": "test_dataset",
            "columns": ["id", "name", "age", "salary"],
            "rows": [
                [1, "Alice", 30, 50000],
                [2, "Bob", 25, 45000],
                [3, "Charlie", 35, 60000]
            ]
        }
        
        response = self.session.post(f"{self.base_url}/api/csv/create", json=payload)
        self.log_test("Create valid CSV", response.status_code == 200, 
                     f"Status: {response.status_code}, Shape: {response.json().get('shape') if response.status_code == 200 else 'N/A'}")
        
        # Empty dataset
        empty_payload = {"name": "empty_dataset"}
        response = self.session.post(f"{self.base_url}/api/csv/create", json=empty_payload)
        self.log_test("Create empty CSV", response.status_code == 200)
        
        # Missing name (should fail)
        invalid_payload = {"columns": ["test"]}
        response = self.session.post(f"{self.base_url}/api/csv/create", json=invalid_payload)
        self.log_test("Create CSV without name", response.status_code == 400)
        
        # Mismatched columns/rows (will expose pandas issues)
        mismatch_payload = {
            "name": "mismatch_test",
            "columns": ["a", "b"],
            "rows": [[1, 2, 3]]  # Too many values
        }
        response = self.session.post(f"{self.base_url}/api/csv/create", json=mismatch_payload)
        self.log_test("Create CSV with column/row mismatch", 
                     response.status_code in [400, 500], 
                     f"Status: {response.status_code}")
    
    def test_load_csv(self):
        """Test CSV file loading"""
        print("\n=== Testing CSV Loading ===")
        
        # Create test CSV content
        csv_content = """id,name,department,salary
1,John Doe,Engineering,75000
2,Jane Smith,Marketing,65000
3,Mike Johnson,Sales,55000
4,Sarah Wilson,HR,60000"""
        
        # Test file upload
        files = {'file': ('test_data.csv', csv_content, 'text/csv')}
        data = {'name': 'uploaded_dataset'}
        response = self.session.post(f"{self.base_url}/api/csv/load", files=files, data=data)
        self.log_test("Load CSV file", response.status_code == 200,
                     f"Loaded shape: {response.json().get('shape') if response.status_code == 200 else 'Failed'}")
        
        # Test without file
        response = self.session.post(f"{self.base_url}/api/csv/load")
        self.log_test("Load CSV without file", response.status_code == 400)
        
        # Test with malformed CSV
        bad_csv = "col1,col2\nvalue1\nvalue2,value3,value4"  # Inconsistent columns
        files = {'file': ('bad.csv', bad_csv, 'text/csv')}
        response = self.session.post(f"{self.base_url}/api/csv/load", files=files)
        # Should handle gracefully but might not
        self.log_test("Load malformed CSV", True, f"Status: {response.status_code}")
    
    def test_dataset_info(self):
        """Test dataset information retrieval"""
        print("\n=== Testing Dataset Info ===")
        
        # Valid dataset
        response = self.session.get(f"{self.base_url}/api/csv/info/test_dataset")
        self.log_test("Get info for existing dataset", response.status_code == 200)
        
        # Non-existent dataset
        response = self.session.get(f"{self.base_url}/api/csv/info/nonexistent")
        self.log_test("Get info for non-existent dataset", response.status_code == 404)
    
    def test_data_retrieval(self):
        """Test data retrieval with pagination"""
        print("\n=== Testing Data Retrieval ===")
        
        # Get full dataset
        response = self.session.get(f"{self.base_url}/api/csv/data/test_dataset")
        if response.status_code == 200:
            total_rows = response.json().get('total_rows', 0)
            self.log_test("Get full dataset", True, f"Total rows: {total_rows}")
        else:
            self.log_test("Get full dataset", False, f"Status: {response.status_code}")
        
        # Test pagination
        response = self.session.get(f"{self.base_url}/api/csv/data/test_dataset?limit=2&offset=1")
        self.log_test("Get paginated data", response.status_code == 200)
        
        # Test invalid limits (negative values, etc.)
        response = self.session.get(f"{self.base_url}/api/csv/data/test_dataset?limit=-1")
        self.log_test("Get data with negative limit", True, f"Status: {response.status_code}")
    
    def test_row_operations(self):
        """Test row CRUD operations"""
        print("\n=== Testing Row Operations ===")
        
        # Add valid row
        new_row = {"row": {"id": 4, "name": "David", "age": 28, "salary": 52000}}
        response = self.session.post(f"{self.base_url}/api/csv/add_row/test_dataset", json=new_row)
        self.log_test("Add valid row", response.status_code == 200)
        
        # Add row with missing columns (pandas will fill with NaN)
        partial_row = {"row": {"id": 5, "name": "Eve"}}
        response = self.session.post(f"{self.base_url}/api/csv/add_row/test_dataset", json=partial_row)
        self.log_test("Add partial row", response.status_code == 200)
        
        # Add row with extra columns (should be ignored or cause issues)
        extra_row = {"row": {"id": 6, "name": "Frank", "age": 40, "salary": 70000, "bonus": 5000}}
        response = self.session.post(f"{self.base_url}/api/csv/add_row/test_dataset", json=extra_row)
        self.log_test("Add row with extra columns", True, f"Status: {response.status_code}")
        
        # Update existing row
        update_data = {"row": {"name": "Alice Updated", "salary": 55000}}
        response = self.session.put(f"{self.base_url}/api/csv/update_row/test_dataset/0", json=update_data)
        self.log_test("Update existing row", response.status_code == 200)
        
        # Update non-existent row
        response = self.session.put(f"{self.base_url}/api/csv/update_row/test_dataset/999", json=update_data)
        self.log_test("Update non-existent row", response.status_code == 400)
        
        # Delete existing row
        response = self.session.delete(f"{self.base_url}/api/csv/delete_row/test_dataset/1")
        self.log_test("Delete existing row", response.status_code == 200)
        
        # Delete non-existent row
        response = self.session.delete(f"{self.base_url}/api/csv/delete_row/test_dataset/999")
        self.log_test("Delete non-existent row", response.status_code == 400)
    
    def test_dataset_management(self):
        """Test dataset-level operations"""
        print("\n=== Testing Dataset Management ===")
        
        # List all datasets
        response = self.session.get(f"{self.base_url}/api/csv/list")
        if response.status_code == 200:
            count = len(response.json().get('datasets', []))
            self.log_test("List datasets", True, f"Found {count} datasets")
        else:
            self.log_test("List datasets", False)
        
        # Export dataset
        response = self.session.get(f"{self.base_url}/api/csv/export/test_dataset")
        self.log_test("Export dataset", response.status_code == 200,
                     f"Content-Type: {response.headers.get('content-type', 'N/A')}")
        
        # Export non-existent dataset
        response = self.session.get(f"{self.base_url}/api/csv/export/nonexistent")
        self.log_test("Export non-existent dataset", response.status_code == 404)
        
        # Delete dataset
        response = self.session.delete(f"{self.base_url}/api/csv/delete/empty_dataset")
        self.log_test("Delete existing dataset", response.status_code == 200)
        
        # Delete non-existent dataset
        response = self.session.delete(f"{self.base_url}/api/csv/delete/nonexistent")
        self.log_test("Delete non-existent dataset", response.status_code == 404)
    
    def test_edge_cases(self):
        """Test edge cases that will break in production"""
        print("\n=== Testing Edge Cases ===")
        
        # Very large dataset name
        long_name = "a" * 1000
        payload = {"name": long_name, "columns": ["test"]}
        response = self.session.post(f"{self.base_url}/api/csv/create", json=payload)
        self.log_test("Create dataset with very long name", True, f"Status: {response.status_code}")
        
        # Dataset name with special characters
        special_name = "test/\\*?<>|"
        payload = {"name": special_name, "columns": ["test"]}
        response = self.session.post(f"{self.base_url}/api/csv/create", json=payload)
        self.log_test("Create dataset with special chars", True, f"Status: {response.status_code}")
        
        # Empty row data
        response = self.session.post(f"{self.base_url}/api/csv/add_row/test_dataset", json={})
        self.log_test("Add row without data", response.status_code == 400)
        
        # Large row count stress test (comment out if you don't want to stress)
        # large_rows = [{"id": i, "name": f"User{i}", "age": 25, "salary": 50000} for i in range(1000)]
        # for i, row in enumerate(large_rows):
        #     response = self.session.post(f"{self.base_url}/api/csv/add_row/test_dataset", json={"row": row})
        #     if i == 10:  # Just test first 10
        #         break
        # self.log_test("Add multiple rows", True, "Partial stress test completed")
    
    def test_concurrent_access(self):
        """Test concurrent access patterns"""
        print("\n=== Testing Concurrent Access ===")
        
        import threading
        import time
        
        def worker(worker_id):
            """Worker thread for concurrent testing"""
            try:
                # Each worker creates its own dataset
                payload = {
                    "name": f"concurrent_test_{worker_id}",
                    "columns": ["id", "worker", "timestamp"],
                    "rows": [[1, worker_id, time.time()]]
                }
                response = self.session.post(f"{self.base_url}/api/csv/create", json=payload)
                return response.status_code == 200
            except Exception as e:
                print(f"Worker {worker_id} failed: {e}")
                return False
        
        # Run concurrent workers
        threads = []
        results = []
        
        for i in range(5):
            thread = threading.Thread(target=lambda i=i: results.append(worker(i)))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        success_count = sum(1 for r in results if r)
        self.log_test("Concurrent dataset creation", success_count == 5, 
                     f"Success rate: {success_count}/5")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("Starting CSV API Test Suite")
        print("=" * 50)
        
        try:
            # Check if server is running
            response = self.session.get(f"{self.base_url}/api/csv/list")
            if response.status_code != 200:
                print(f"❌ Server not responding at {self.base_url}")
                return
        
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to server at {self.base_url}")
            print("Make sure your Flask app is running with: python your_api.py")
            return
        
        self.test_create_csv()
        self.test_load_csv()
        self.test_dataset_info()
        self.test_data_retrieval()
        self.test_row_operations()
        self.test_dataset_management()
        self.test_edge_cases()
        self.test_concurrent_access()
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for test in self.test_results if test["success"])
        total = len(self.test_results)
        
        print(f"Passed: {passed}/{total}")
        print(f"Failed: {total - passed}/{total}")
        
        if total - passed > 0:
            print("\nFailed tests:")
            for test in self.test_results:
                if not test["success"]:
                    print(f"  - {test['test']}: {test['details']}")

def main():
    base_url = "http://localhost:5001"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    tester = CSVAPITester(base_url)
    tester.run_all_tests()

if __name__ == "__main__":
    main()