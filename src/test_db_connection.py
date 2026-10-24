import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


# Find and load the .env file in the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def test_database_connection():
    """Test the connection to the PostgreSQL database."""

    connection = None

    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=5,
        )

        print("SUCCESS: Connected to PostgreSQL!")

        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), version();")
            database_name, version = cursor.fetchone()

            print(f"Database: {database_name}")
            print(f"PostgreSQL version: {version.split(',')[0]}")

    except psycopg2.Error as error:
        print("ERROR: Could not connect to PostgreSQL.")
        print(f"Details: {error}")

    finally:
        if connection is not None:
            connection.close()
            print("Database connection closed.")


if __name__ == "__main__":
    test_database_connection()