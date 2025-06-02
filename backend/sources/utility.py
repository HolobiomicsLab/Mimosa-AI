
from termcolor import colored
import platform
import threading

thinking_event = threading.Event()
current_animation_thread = None

def get_color_map():
    if platform.system().lower() != "windows":
        color_map = {
            "success": "green",
            "failure": "red",
            "status": "light_green",
            "code": "light_blue",
            "warning": "yellow",
            "output": "cyan",
            "info": "cyan"
        }
    else:
        color_map = {
            "success": "green",
            "failure": "red",
            "status": "light_green",
            "code": "light_blue",
            "warning": "yellow",
            "output": "cyan",
            "info": "black"
        }
    return color_map

def pretty_print(text, color="info", no_newline=False):
    """
    Print text with color formatting.

    Args:
        text (str): The text to print
        color (str, optional): The color to use. Defaults to "info".
            Valid colors are:
            - "success": Green
            - "failure": Red 
            - "status": Light green
            - "code": Light blue
            - "warning": Yellow
            - "output": Cyan
            - "default": Black (Windows only)
    """
    thinking_event.set()
    if current_animation_thread and current_animation_thread.is_alive():
        current_animation_thread.join()
    thinking_event.clear()
    
    color_map = get_color_map()
    if color not in color_map:
        color = "info"
    print(colored(text, color_map[color]), end='' if no_newline else "\n")

def timer_decorator(func):
    """
    Decorator to measure the execution time of a function.
    Usage:
    @timer_decorator
    def my_function():
        # code to execute
    """
    from time import time
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        pretty_print(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute", "status")
        return result
    return wrapper