"""
Enhanced Biometric API Service with detailed logging and proper error handling
Handles communication with the biometric device API
"""

import aiohttp
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, date, timedelta
from fastapi import HTTPException
import logging
from time import time

logger = logging.getLogger(__name__)

class BiometricAPIService:
    def __init__(self):
        self.base_url = "https://hr.iuea.app"
        self.username = "admin"
        self.password = "adm1N2022"
        self.token = None
        self.session = None
        
        # Rate limiting settings
        self.delay_between_calls = 3.0
        self.max_retries = 3
        self.request_timeout = 60
        self.last_call_time = 0
    
    async def __aenter__(self):
        """Initialize session and authenticate"""
        try:
            logger.info("🔧 Initializing BiometricAPIService...")
            
            # Create session with working settings
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            # Use the same SSL settings that worked in the test
            connector = aiohttp.TCPConnector(
                ssl=False,  # This worked in the test
                limit=10,
                limit_per_host=5
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'HR-Attendance-System/1.0',
                    'Accept': 'application/json'
                }
            )
            
            # Authenticate using the same method that worked in test
            logger.info("🔐 Attempting authentication...")
            success = await self._authenticate()
            if not success:
                await self.session.close()
                raise HTTPException(status_code=401, detail="Authentication failed")
            
            logger.info(f"✅ BiometricAPIService initialized successfully")
            return self
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize BiometricAPIService: {e}")
            if self.session and not self.session.closed:
                await self.session.close()
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("🔒 BiometricAPIService session closed")
    
    async def _authenticate(self) -> bool:
        """Authenticate with the API - using the same method that worked in test"""
        auth_url = f"{self.base_url}/api-token-auth/"
        
        try:
            logger.info(f"📡 POST {auth_url}")
            logger.info(f"📝 Username: {self.username}")
            
            # Use the exact same approach that worked in the test
            async with self.session.post(
                auth_url,
                json={'username': self.username, 'password': self.password}
            ) as response:
                
                logger.info(f"📊 Authentication response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    self.token = data.get('token')
                    if self.token:
                        logger.info(f"🎫 Token received: {self.token[:20]}...")
                        return True
                    else:
                        logger.error("❌ No token received in response")
                        logger.debug(f"Response data: {data}")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Authentication failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"💥 Authentication error: {type(e).__name__}: {e}")
            return False
    
    async def _rate_limit_delay(self):
        """Simple rate limiting to avoid overwhelming the API"""
        current_time = time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.delay_between_calls:
            delay = self.delay_between_calls - time_since_last_call
            logger.debug(f"⏳ Rate limiting: waiting {delay:.1f} seconds")
            await asyncio.sleep(delay)
        
        self.last_call_time = time()
    
    async def fetch_attendance_data(
        self,
        emp_code: str,
        start_date: date,
        end_date: Optional[date] = None
    ) -> List[Dict]:
        """Fetch attendance data for specific employee with detailed logging"""
        if not self.token:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        if not end_date:
            end_date = start_date
        
        logger.info("=" * 80)
        logger.info(f"🔍 FETCHING ATTENDANCE DATA")
        logger.info(f"📋 Employee Code: '{emp_code}'")
        logger.info(f"📅 Date Range: {start_date} to {end_date}")
        logger.info("=" * 80)
        
        headers = {'Authorization': f'Token {self.token}'}
        endpoint = f"{self.base_url}/iclock/api/transactions/"
        
        all_records = []
        page = 1
        
        try:
            while True:
                # Apply rate limiting
                await self._rate_limit_delay()
                
                params = {
                    'emp_code': emp_code,
                    'start_time': start_date.isoformat(),
                    'end_time': end_date.isoformat(),
                    'page': page,
                    'page_size': 100,
                    'order_by': 'punch_time'
                }
                
                logger.info(f"📄 Fetching page {page}...")
                logger.info(f"🔗 URL: {endpoint}")
                logger.info(f"📝 Params: {params}")
                
                # Check if session is still open
                if self.session.closed:
                    raise HTTPException(status_code=500, detail="Session closed unexpectedly")
                
                async with self.session.get(endpoint, headers=headers, params=params) as response:
                    logger.info(f"📊 Response Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        records = data.get('data', [])
                        
                        logger.info(f"📦 Page {page} Records: {len(records)}")
                        
                        # Log sample records for debugging
                        if records:
                            logger.info(f"📋 Sample record from page {page}:")
                            logger.info(f"   {records[0]}")
                        
                        all_records.extend(records)
                        
                        # Check pagination
                        has_next = data.get('next')
                        total_count = data.get('count', 0)
                        logger.info(f"🔢 Total available records: {total_count}")
                        logger.info(f"➡️ Has next page: {bool(has_next)}")
                        
                        if not has_next:
                            break
                        page += 1
                        
                    elif response.status == 401:
                        logger.error("🔐 Authentication token expired")
                        raise HTTPException(status_code=401, detail="Token expired")
                    elif response.status == 404:
                        logger.warning(f"🔍 No data found for employee code: {emp_code}")
                        break
                    else:
                        error_text = await response.text()
                        logger.error(f"💥 API request failed: {response.status}")
                        logger.error(f"📄 Error response: {error_text}")
                        break
            
            logger.info("=" * 80)
            logger.info(f"✅ FETCH COMPLETED")
            logger.info(f"📊 Total records fetched: {len(all_records)}")
            if all_records:
                logger.info(f"⏰ Date range in data: {min(r['punch_time'] for r in all_records)} to {max(r['punch_time'] for r in all_records)}")
            else:
                logger.warning(f"⚠️ No attendance records found for employee '{emp_code}' in the specified date range")
                logger.info(f"💡 This could mean:")
                logger.info(f"   - Employee didn't punch during this period")
                logger.info(f"   - Biometric ID '{emp_code}' doesn't exist in the system")
                logger.info(f"   - Date range {start_date} to {end_date} has no data")
            logger.info("=" * 80)
            
            return all_records
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"💥 Error fetching attendance data: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch attendance data: {str(e)}")
    
    async def fetch_all_recent_data(self, days_back: int = 1) -> List[Dict]:
        """Fetch all recent attendance data for testing purposes"""
        if not self.token:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        headers = {'Authorization': f'Token {self.token}'}
        endpoint = f"{self.base_url}/iclock/api/transactions/"
        
        all_records = []
        page = 1
        
        logger.info(f"🔍 Fetching bulk data for last {days_back} days")
        
        try:
            while True:
                await self._rate_limit_delay()
                
                params = {
                    'start_time': start_date.isoformat(),
                    'end_time': end_date.isoformat(),
                    'page': page,
                    'page_size': 100,
                    'order_by': 'punch_time'
                }
                
                if self.session.closed:
                    raise HTTPException(status_code=500, detail="Session closed unexpectedly")
                
                async with self.session.get(endpoint, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        records = data.get('data', [])
                        all_records.extend(records)
                        
                        if not data.get('next'):
                            break
                        page += 1
                    else:
                        logger.error(f"Bulk fetch failed: {response.status}")
                        break
            
            logger.info(f"✅ Bulk fetch completed: {len(all_records)} records")
            return all_records
            
        except Exception as e:
            logger.error(f"Error in bulk fetch: {e}")
            return []
    
    def parse_attendance_record(self, api_record: Dict) -> Optional[Dict]:
        """Parse API record into database format with enhanced error handling"""
        try:
            logger.debug(f"🔄 Parsing record ID: {api_record.get('id')}")
            
            # Parse punch time
            punch_time_str = api_record.get('punch_time')
            if not punch_time_str:
                logger.error(f"❌ No punch_time in record: {api_record}")
                return None
            
            try:
                punch_time = datetime.strptime(punch_time_str, '%Y-%m-%d %H:%M:%S')
            except ValueError as ve:
                logger.error(f"❌ Invalid punch_time format '{punch_time_str}': {ve}")
                return None
            
            # Parse upload time (optional)
            upload_time = None
            upload_time_str = api_record.get('upload_time')
            if upload_time_str:
                try:
                    upload_time = datetime.strptime(upload_time_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    logger.warning(f"⚠️ Invalid upload_time format '{upload_time_str}', setting to None")
            
            parsed = {
                'biometric_record_id': api_record.get('id'),
                'emp_code': api_record.get('emp_code'),
                'punch_time': punch_time,
                'punch_state': api_record.get('punch_state'),
                'punch_state_display': api_record.get('punch_state_display'),
                'verify_type': api_record.get('verify_type'),
                'verify_type_display': api_record.get('verify_type_display'),
                'terminal_sn': api_record.get('terminal_sn'),
                'terminal_alias': api_record.get('terminal_alias'),
                'area_alias': api_record.get('area_alias'),
                'temperature': float(api_record.get('temperature', 0.0)),
                'upload_time': upload_time
            }
            
            logger.debug(f"✅ Parsed: {punch_time} - {api_record.get('punch_state_display', 'Unknown')}")
            return parsed
            
        except Exception as e:
            logger.error(f"💥 Error parsing record {api_record.get('id')}: {e}")
            logger.error(f"📄 Raw record: {api_record}")
            return None
    
    async def test_connection(self) -> Dict:
        """Test the API connection and return status"""
        try:
            if not self.token:
                return {
                    'success': False,
                    'message': 'Not authenticated',
                    'api_status': 'Disconnected'
                }
            
            # Test with a simple request
            headers = {'Authorization': f'Token {self.token}'}
            endpoint = f"{self.base_url}/iclock/api/transactions/"
            
            params = {
                'page': 1,
                'page_size': 1
            }
            
            async with self.session.get(endpoint, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'success': True,
                        'message': 'API connection successful',
                        'api_status': 'Connected',
                        'total_records': data.get('count', 0),
                        'test_timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'message': f'API returned status {response.status}',
                        'api_status': 'Error'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection test failed: {str(e)}',
                'api_status': 'Failed'
            }