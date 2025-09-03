import os
import json
from datetime import datetime

class ValidationHandler:
    def __init__(self):
        self.upload_folder = os.path.join('static', 'uploads')
        self.database_file = os.path.join(self.upload_folder, 'database.txt')
        self._ensure_database_exists()

    def _ensure_database_exists(self):
        """Make sure database file exists and is valid JSON"""
        if not os.path.exists(self.database_file):
            # Create empty database
            self.save_database([])
        else:
            try:
                # Verify it's valid JSON
                with open(self.database_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError:
                # Backup corrupted file and create new one
                backup_name = f"database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                os.rename(self.database_file, os.path.join(self.upload_folder, backup_name))
                self.save_database([])

    def load_database(self):
        """Load the database from file with error handling"""
        try:
            with open(self.database_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            return []

    def save_database(self, data):
        """Save the database with proper formatting and error handling"""
        try:
            # Create temp file
            temp_file = self.database_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            # Rename temp file to actual file
            if os.path.exists(self.database_file):
                os.remove(self.database_file)
            os.rename(temp_file, self.database_file)
            return True
        except Exception as e:
            print(f"Error saving database: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def update_valid_status(self, image_id, place):
        """Update valid_status for a specific crop with improved error handling"""
        try:
            # Load current database
            database = self.load_database()
            if not database:
                return False, "Could not load database"

            # Find and update the specific crop entry
            found = False
            updated = False
            
            for entry in database:
                if entry['file_name'].startswith(image_id) and entry.get('place') == place:
                    found = True
                    # Only update if not already validated
                    if not entry.get('valid_status', False):
                        entry['valid_status'] = True
                        updated = True
                    break

            if not found:
                return False, f"No matching entry found for image_id: {image_id} and place: {place}"
            
            if not updated:
                return False, "Entry already validated"

            # Save the updated database
            if self.save_database(database):
                return True, "Valid status updated successfully"
            else:
                return False, "Failed to save changes to database"

        except Exception as e:
            print(f"Error in update_valid_status: {str(e)}")
            return False, f"Internal error: {str(e)}"
