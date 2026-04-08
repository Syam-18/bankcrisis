from bankcrisis.server.app import app as real_app
from bankcrisis.server.app import main as real_main

app = real_app


def main():
    real_main()


if __name__ == "__main__":
    main()