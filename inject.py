# create_admin_user.py

import sqlite3
import os
from datetime import datetime
from passlib.context import CryptContext

# Initialize password context (same as in your auth system)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_admin_user():
    db_path = "hr_system.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        print("Make sure you're running this script in the same directory as hr_system.db")
        return False
    
    print(f"🔍 Found database: {db_path}")
    
    try:
        # Create backup first
        backup_path = f"hr_system_backup_admin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        print(f"📁 Creating backup: {backup_path}")
        
        with open(db_path, 'rb') as original:
            with open(backup_path, 'wb') as backup:
                backup.write(original.read())
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if users table exists
        print("🔍 Checking if users table exists...")
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='users'
        """)
        
        existing_table = cursor.fetchone()
        
        if not existing_table:
            print("❌ Users table does not exist!")
            print("Creating users table...")
            
            # Create users table if it doesn't exist
            cursor.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(255),
                    hashed_password VARCHAR(255) NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✅ Users table created!")
        else:
            print("✅ Users table found!")
        
        # Check if admin user already exists
        print("\n🔍 Checking if admin user already exists...")
        cursor.execute("SELECT username FROM users WHERE username = ?", ("admin",))
        existing_admin = cursor.fetchone()
        
        if existing_admin:
            print("⚠️  Admin user already exists!")
            
            # Ask if user wants to update the password
            update_choice = input("Do you want to update the admin password? (y/N): ").lower().strip()
            
            if update_choice in ['y', 'yes']:
                # Hash the new password
                hashed_password = pwd_context.hash("mansuronly")
                
                # Update admin password
                cursor.execute("""
                    UPDATE users 
                    SET hashed_password = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE username = ?
                """, (hashed_password, "admin"))
                
                print("✅ Admin password updated successfully!")
            else:
                print("ℹ️  Admin user left unchanged.")
                return True
        else:
            # Hash the password
            print("🔐 Hashing password...")
            hashed_password = pwd_context.hash("mansuronly")
            
            # Create admin user
            print("👤 Creating admin user...")
            cursor.execute("""
                INSERT INTO users (username, email, hashed_password, is_active, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "admin",
                "admin@hrsystem.local",
                hashed_password,
                True,
                datetime.utcnow()
            ))
            
            print("✅ Admin user created successfully!")
        
        # Commit changes
        conn.commit()
        
        # Verify the admin user
        print("\n🔍 Verifying admin user...")
        cursor.execute("""
            SELECT id, username, email, is_active, created_at 
            FROM users WHERE username = ?
        """, ("admin",))
        
        admin_user = cursor.fetchone()
        
        if admin_user:
            print("✅ Admin user verification successful!")
            print(f"   ID: {admin_user[0]}")
            print(f"   Username: {admin_user[1]}")
            print(f"   Email: {admin_user[2]}")
            print(f"   Active: {admin_user[3]}")
            print(f"   Created: {admin_user[4]}")
        else:
            print("❌ Admin user verification failed!")
            return False
        
        # Test password verification
        print("\n🔍 Testing password verification...")
        cursor.execute("SELECT hashed_password FROM users WHERE username = ?", ("admin",))
        stored_hash = cursor.fetchone()[0]
        
        if pwd_context.verify("mansuronly", stored_hash):
            print("✅ Password verification successful!")
        else:
            print("❌ Password verification failed!")
            return False
        
        # Show total users count
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        print(f"\n📊 Total users in system: {total_users}")
        
        print(f"\n🎉 SUCCESS! Admin user setup completed.")
        print(f"📁 Backup saved as: {backup_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to show more details about the error
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()

def verify_admin_login():
    """Verify admin can login with the created credentials"""
    try:
        conn = sqlite3.connect("hr_system.db")
        cursor = conn.cursor()
        
        print("\n🔍 Final login verification test...")
        
        # Get admin user
        cursor.execute("""
            SELECT username, hashed_password, is_active 
            FROM users WHERE username = ?
        """, ("admin",))
        
        admin_data = cursor.fetchone()
        
        if not admin_data:
            print("❌ Admin user not found!")
            return False
        
        username, hashed_password, is_active = admin_data
        
        # Check if user is active
        if not is_active:
            print("❌ Admin user is not active!")
            return False
        
        # Verify password
        if pwd_context.verify("mansuronly", hashed_password):
            print("✅ Login credentials verified successfully!")
            print(f"   Username: {username}")
            print(f"   Password: mansuronly")
            print(f"   Status: Active")
            return True
        else:
            print("❌ Password verification failed!")
            return False
        
    except Exception as e:
        print(f"❌ Login verification failed: {e}")
        return False
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🚀 HR System Admin User Creation")
    print("=" * 60)
    print("Creating admin user with credentials:")
    print("   Username: admin")
    print("   Password: mansuronly")
    print()
    
    success = create_admin_user()
    
    if success:
        verify_admin_login()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ADMIN USER SETUP COMPLETED SUCCESSFULLY!")
        print()
        print("✅ Admin user credentials:")
        print("   👤 Username: admin")
        print("   🔐 Password: mansuronly")
        print("   📧 Email: admin@hrsystem.local")
        print("   ✔️  Status: Active")
        print()
        print("🔗 You can now:")
        print("   ✅ Login to the HR system using these credentials")
        print("   ✅ Access all administrative functions")
        print("   ✅ Manage other users through the users page")
        print("   ✅ Configure system settings")
        print()
        print("📋 Next steps:")
        print("1. Start/restart your FastAPI application")
        print("2. Navigate to the login page")
        print("3. Use the admin credentials to login")
        print("4. Consider changing the password after first login")
        print("5. The backup file has been created for safety")
        print()
        print("⚠️  Security Note:")
        print("   For production systems, please change the default")
        print("   admin password to something more secure!")
    else:
        print("💥 ADMIN USER CREATION FAILED!")
        print()
        print("Please check the error messages above and try again.")
        print("Your original database remains unchanged.")
        print()
        print("🔧 Troubleshooting:")
        print("1. Make sure hr_system.db is not locked by another process")
        print("2. Ensure you have write permissions to the directory")
        print("3. Check that the database file is not corrupted")
        print("4. Make sure no other applications are using the database")
        print("5. Verify that the passlib library is installed (pip install passlib[bcrypt])")
