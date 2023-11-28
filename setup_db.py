try:
    # At the top of your start.py script, after importing necessary modules
    import psycopg2
    import getpass
    import os
    import subprocess
    import sys
    import importlib

    print("Import Success")
except:
    print("Import Error")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

current_directory = os.path.dirname(os.path.abspath(__file__))

required_libraries = [
    "subprocess",
    "psycopg2",
    "getpass",
    "os",
    "importlib",
    "sys",
    # any other libraries you want to ensure are installed
]

missing_libraries = []

for library in required_libraries:
    try:
        importlib.import_module(library)
        print(f"{library} is installed.")
    except ImportError:
        print(f"{library} is not installed.")
        missing_libraries.append(library)

if missing_libraries:
    print("Attempting to install missing libraries...")
    subprocess.run(["pip", "install", *missing_libraries])
    print("Installation of missing libraries completed.")

def run_psql_command(command, env):
    try:
        result = subprocess.run(['psql', '-c', command], env=env, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")
        return None

def create_role_and_database(superuser, superuser_password, role_name, role_password, db_name):
    try:
        env = {'PGUSER': superuser, 'PGPASSWORD': superuser_password}
        
        # Check if role already exists
        role_check = run_psql_command(f"SELECT 1 FROM pg_roles WHERE rolname='{role_name}';", env)
        if role_check and '1 row' in role_check:
            print(f"Role {role_name} already exists.")
        else:
            # Create role
            run_psql_command(f"CREATE ROLE {role_name} WITH LOGIN PASSWORD '{role_password}';", env)
            print(f"Role {role_name} created.")

        # Check if database already exists
        db_check = run_psql_command(f"SELECT 1 FROM pg_database WHERE datname='{db_name}';", env)
        if db_check and '1 row' in db_check:
            print(f"Database {db_name} already exists.")
        else:
            # Create database
            run_psql_command(f"CREATE DATABASE {db_name};", env)
            run_psql_command(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {role_name};", env)
            print(f"Database {db_name} created and privileges granted to {role_name}.")

    except Exception as e:
        print(f"An error occurred during database setup: {e}")

def write_db_config(db_name, role_name, role_password):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    with open(f'{current_directory}/chemical_databases/db_config.txt', 'w') as file:
        file.write(f"dbname={db_name}\n")
        file.write(f"user={role_name}\n")
        file.write(f"password={role_password}\n")
        file.write("host=localhost\n")
    print("Database configuration file written.")


def detect_os():
    if sys.platform.startswith('darwin'):
        return 'macOS'
    elif sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
        return 'Windows'
    elif sys.platform.startswith('linux'):
        return 'Linux'
    else:
        return 'Unknown'

def is_postgresql_installed():
    try:
        subprocess.run(['psql', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def main():
    print("Welcome to the Database Setup for Frank's Chemical Process Simulator")

    # Detect OS and check PostgreSQL installation
    os_type = detect_os()
    print(f"Detected operating system: {os_type}")

    if not is_postgresql_installed():
        print("PostgreSQL is not installed or not added to PATH. Please install PostgreSQL and ensure 'psql' is accessible in your PATH.")
        if os_type == 'macOS':
            print("Attempting to add PostgreSQL to PATH for this session.")
            os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
            if is_postgresql_installed():
                print("PostgreSQL found after updating PATH.")
            else:
                print("PostgreSQL still not found. Please ensure it's installed and '/opt/homebrew/bin' is in your PATH.")
                return
            print("On macOS, you can install PostgreSQL using Homebrew:")
            print("  brew install postgresql")
            print("  brew services start postgresql")
            print("After installation, ensure that 'psql' is in your PATH.")
        elif os_type == 'Linux':
            print("On Linux, you can install PostgreSQL using your distribution's package manager.")
            print("For example, on Ubuntu:")
            print("  sudo apt-get update")
            print("  sudo apt-get install postgresql postgresql-contrib")
        elif os_type == 'Windows':
            print("On Windows, download and install PostgreSQL from the official website.")
            print("Ensure that the installation path is added to your system's PATH.")
        return

    print("Welcome to the Database Setup for Frank's Chemical Process Simulator")
    
    # Prompt for PostgreSQL superuser username and password
    superuser = input("Enter your PostgreSQL superuser username (typically 'postgres'): ")
    superuser_password = getpass.getpass("Enter your PostgreSQL superuser password: ")

    # Prompt for new role and database details
    role_name = input("Enter a new role name for the application (e.g., 'User'): ")
    role_password = getpass.getpass("Enter a password for the new role: ")
    db_name = input("Enter the name for the new database (e.g., 'DATA'): ")

    create_role_and_database(superuser, superuser_password, role_name, role_password, db_name)
    write_db_config(db_name, role_name, role_password)

if __name__ == "__main__":
    main()