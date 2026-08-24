import datetime as dt
import os


def get_time():
    return dt.datetime.now().strftime("%I:%M:%S %p")


def get_date():
    return dt.datetime.now().strftime("%A, %B %d, %Y")


def get_current_directory():
    return os.getcwd()


if __name__ == "__main__":
    print("Time:", get_time())
    print("Date:", get_date())
    print("Directory:", get_current_directory())
