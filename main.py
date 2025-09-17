from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os

def main():
    print("Hello from cv-generator!")
    print("Environment Variables:")
    for key, value in os.environ.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
