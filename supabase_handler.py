import os
from supabase import create_client, Client
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

class SupabaseHandler:
    def __init__(self):
        # Supabase configuration - replace with your actual values
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_ANON_KEY')
        self.supabase: Client = create_client(self.url, self.key)
        
        # Table name
        self.table_name = 'palm_detections'

    def add_detection(self, file_name: str, class_result: str, place, uid: str = "default_user", valid_status: bool = False) -> bool:
        """Add a new detection record to Supabase"""
        try:
            # Prepare data
            data = {
                'file_name': file_name,
                'uid': uid,
                'class_result': class_result,
                'valid_status': valid_status,
                'place': str(place) if place == "-" else place,
                'created_at': datetime.now().isoformat()
            }
            
            # Insert into Supabase
            result = self.supabase.table(self.table_name).insert(data).execute()
            
            if result.data:
                print(f"Successfully added detection: {file_name}")
                return True
            else:
                print(f"Failed to add detection: {file_name}")
                return False
                
        except Exception as e:
            print(f"Error adding detection: {str(e)}")
            return False

    def get_all_detections(self) -> List[Dict]:
        """Get all detection records from Supabase"""
        try:
            result = self.supabase.table(self.table_name).select("*").execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting detections: {str(e)}")
            return []

    def get_detections_by_date(self, date: str) -> List[Dict]:
        """Get detections for a specific date (YYYY-MM-DD format)"""
        try:
            # Convert date to start and end of day
            start_date = f"{date}T00:00:00"
            end_date = f"{date}T23:59:59"
            
            result = self.supabase.table(self.table_name)\
                .select("*")\
                .gte('created_at', start_date)\
                .lte('created_at', end_date)\
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting detections by date: {str(e)}")
            return []

    def get_detections_by_base_name(self, base_name: str) -> List[Dict]:
        """Get all detections for a specific image set (by base filename)"""
        try:
            result = self.supabase.table(self.table_name)\
                .select("*")\
                .like('file_name', f"{base_name}%")\
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting detections by base name: {str(e)}")
            return []

    def update_valid_status(self, file_name: str, place, valid_status: bool = True) -> Tuple[bool, str]:
        """Update validation status for a specific detection"""
        try:
            # Find the specific record
            result = self.supabase.table(self.table_name)\
                .select("*")\
                .eq('file_name', file_name)\
                .eq('place', place)\
                .execute()
            
            if not result.data:
                return False, f"No matching record found for {file_name} with place {place}"
            
            record = result.data[0]
            
            # Check if already validated
            if record.get('valid_status', False):
                return False, "Record already validated"
            
            # Update the record
            update_result = self.supabase.table(self.table_name)\
                .update({'valid_status': valid_status})\
                .eq('id', record['id'])\
                .execute()
            
            if update_result.data:
                return True, "Valid status updated successfully"
            else:
                return False, "Failed to update valid status"
                
        except Exception as e:
            print(f"Error updating valid status: {str(e)}")
            return False, f"Internal error: {str(e)}"

    def get_class_summary(self) -> Dict[str, int]:
        """Get summary of all classifications"""
        try:
            detections = self.get_all_detections()
            class_counts = {}
            
            for detection in detections:
                if detection['place'] not in ['-', 0, '0']:  # Only count actual detections
                    class_name = detection['class_result']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            return class_counts
        except Exception as e:
            print(f"Error getting class summary: {str(e)}")
            return {}

    def delete_detection(self, detection_id: int) -> bool:
        """Delete a specific detection record"""
        try:
            result = self.supabase.table(self.table_name)\
                .delete()\
                .eq('id', detection_id)\
                .execute()
            
            return bool(result.data)
        except Exception as e:
            print(f"Error deleting detection: {str(e)}")
            return False

    def get_detections_by_uid(self, uid: str) -> List[Dict]:
        """Get detections for a specific user"""
        try:
            result = self.supabase.table(self.table_name)\
                .select("*")\
                .eq('uid', uid)\
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting detections by UID: {str(e)}")
            return []