#!/usr/bin/env python3
"""
Improved biometric API test - trying different datetime formats
"""

import requests
import json
from datetime import datetime, date, timezone
import sys

# Configuration
BASE_URL = "https://hr.iuea.app"
USERNAME = "admin"
PASSWORD = "adm1N2022"

class BiometricAPITester:
    def __init__(self):
        self.base_url = BASE_URL.rstrip('/')
        self.token = None
        self.session = requests.Session()
        
    def authenticate(self):
        """Get authentication token"""
        auth_url = f"{self.base_url}/api-token-auth/"
        
        response = self.session.post(
            auth_url, 
            json={'username': USERNAME, 'password': PASSWORD}
        )
        
        if response.status_code == 200:
            self.token = response.json()['token']
            print(f"✅ Authentication successful")
            return True
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(response.text)
            return False
    
    def test_endpoint_without_filters(self):
        """Test the endpoint without any filters to see basic structure"""
        print(f"\n🔍 Testing endpoint without filters...")
        
        headers = {'Authorization': f'Token {self.token}'}
        
        response = self.session.get(
            f"{self.base_url}/iclock/api/transactions/",
            headers=headers,
            params={'page_size': 5}  # Just get 5 records to see structure
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Got response:")
            print(json.dumps(data, indent=2, default=str))
            return data
        else:
            print(f"❌ Failed:")
            print(response.text)
            return None
    
    def test_different_date_formats(self, emp_code="00189"):
        """Test different datetime formats"""
        headers = {'Authorization': f'Token {self.token}'}
        endpoint = f"{self.base_url}/iclock/api/transactions/"
        
        # Different date formats to try
        test_date = date.today()
        formats_to_try = [
            # Format 1: ISO with timezone
            {
                'name': 'ISO with Z timezone',
                'start_time': f'{test_date}T00:00:00Z',
                'end_time': f'{test_date}T23:59:59Z'
            },
            # Format 2: ISO without timezone
            {
                'name': 'ISO without timezone',
                'start_time': f'{test_date}T00:00:00',
                'end_time': f'{test_date}T23:59:59'
            },
            # Format 3: Just date
            {
                'name': 'Date only',
                'start_time': str(test_date),
                'end_time': str(test_date)
            },
            # Format 4: Django datetime format
            {
                'name': 'Django format',
                'start_time': f'{test_date} 00:00:00',
                'end_time': f'{test_date} 23:59:59'
            },
            # Format 5: Different ISO format
            {
                'name': 'ISO with +00:00',
                'start_time': f'{test_date}T00:00:00+00:00',
                'end_time': f'{test_date}T23:59:59+00:00'
            },
            # Format 6: Unix timestamp
            {
                'name': 'Unix timestamp',
                'start_time': str(int(datetime.combine(test_date, datetime.min.time()).timestamp())),
                'end_time': str(int(datetime.combine(test_date, datetime.max.time()).timestamp()))
            }
        ]
        
        for fmt in formats_to_try:
            print(f"\n🧪 Testing format: {fmt['name']}")
            print(f"   start_time: {fmt['start_time']}")
            print(f"   end_time: {fmt['end_time']}")
            
            params = {
                'emp_code': emp_code,
                'start_time': fmt['start_time'],
                'end_time': fmt['end_time'],
                'page_size': 10
            }
            
            response = self.session.get(endpoint, headers=headers, params=params)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ SUCCESS!")
                print(f"   Records found: {len(data.get('data', []))}")
                if data.get('data'):
                    print(f"   Sample record: {data['data'][0]}")
                return data, fmt
            else:
                error_msg = response.text[:100] + "..." if len(response.text) > 100 else response.text
                print(f"   ❌ Failed: {error_msg}")
        
        return None, None
    
    def test_employee_filter_only(self, emp_code="00189"):
        """Test with just employee filter, no date"""
        print(f"\n👤 Testing with just employee filter: {emp_code}")
        
        headers = {'Authorization': f'Token {self.token}'}
        
        params = {
            'emp_code': emp_code,
            'page_size': 10,
            'order_by': '-punch_time'
        }
        
        response = self.session.get(
            f"{self.base_url}/iclock/api/transactions/",
            headers=headers,
            params=params
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Found {len(data.get('data', []))} records")
            if data.get('data'):
                print("Recent records:")
                for i, record in enumerate(data['data'][:3]):  # Show first 3
                    print(f"  {i+1}. {record}")
            return data
        else:
            print(f"❌ Failed: {response.text}")
            return None
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🚀 Comprehensive Biometric API Test")
        print("=" * 60)
        
        # Step 1: Authenticate
        if not self.authenticate():
            return
        
        # Step 2: Test endpoint without filters
        basic_data = self.test_endpoint_without_filters()
        
        # Step 3: Test employee filter only
        employee_data = self.test_employee_filter_only("00189")
        
        # Step 4: Test different date formats
        if employee_data:  # Only if employee exists
            success_data, working_format = self.test_different_date_formats("00189")
            
            if working_format:
                print(f"\n🎉 WORKING DATE FORMAT FOUND: {working_format['name']}")
                print(f"   Use this format for integration:")
                print(f"   start_time: {working_format['start_time']}")
                print(f"   end_time: {working_format['end_time']}")
        
        # Step 5: Summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Authentication: {'Success' if self.token else 'Failed'}")
        print(f"✅ Basic endpoint: {'Success' if basic_data else 'Failed'}")
        print(f"✅ Employee 00189: {'Found' if employee_data else 'Not found'}")
        
        if basic_data and 'data' in basic_data and basic_data['data']:
            sample = basic_data['data'][0]
            print(f"\n📋 Sample Record Structure:")
            for key, value in sample.items():
                print(f"   {key}: {type(value).__name__} = {value}")

def main():
    tester = BiometricAPITester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()