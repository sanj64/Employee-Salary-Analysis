# --- src/utils.py ---
# This file is for any general utility functions or helper methods
# that don't fit into the other specific categories.
# For this project, you might not strictly need it unless you add
# custom logging, plotting functions, or other reusable utilities.

# Example (you can leave this file empty if not needed, or add functions as you go):
def print_separator(char='-', length=50):
    """Prints a separator line for better output readability."""
    print(char * length)

def custom_logger(message, level="INFO"):
    """A simple custom logger function."""
    print(f"[{level}] {message}")