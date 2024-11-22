import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np

def select_sheet_path():
    """
    Opens a file dialog to select an Excel or LibreOffice Calc file and returns its path.

    :return: The file path of the selected Excel or LibreOffice Calc file.
    :rtype: str
    """

    # Create the hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open the file selection dialog to choose an Excel file
    file_path = filedialog.askopenfilename(
        title="Select an Excel spreadsheet",
        filetypes=[("Excel/LibreOffice Calc Files", "*.xlsx *.xls *.odf")]
    )

    # Return the path of the selected file
    return file_path


def read_c0(path):
    """
    Transformms an Excel sheet into a NumPy array.

    :param path: Path to the Excel sheet containing the initial condition data.
    :type path: str
    :return: Array containing the initial conditions from the specified sheet.
    :rtype: numpy.ndarray
    :raises FileNotFoundError: If the file is not selected
    """

    sheet = pd.read_excel(f'{path}', sheet_name='Condição Inicial')
    sheet.drop(sheet.columns[0], axis=1, inplace=True)
    sheet = sheet.iloc[:, [1, 0, 2, 3]]
    sheet = sheet.to_numpy()

    return sheet



def read_c0_columns(path):
    """
    Transforms an Excel sheet into separate NumPy arrays for each column.

    :param path: Path to the Excel sheet containing the initial condition data.
    :type path: str
    :return: Tuple of arrays, each containing the data of one column from the specified sheet.
    :rtype: tuple of numpy.ndarray
    :raises FileNotFoundError: If the file is not found.
    """
    # Read the specified sheet from the Excel file
    sheet = pd.read_excel(path, sheet_name='Condição Inicial')

    # Drop the first column
    sheet.drop(sheet.columns[0], axis=1, inplace=True)

    # Reorder columns
    sheet = sheet.iloc[:, [1, 0, 2, 3]]

    # Convert to NumPy array
    sheet_array = sheet.to_numpy()

    # Split the array into individual columns
    names = sheet_array[:, 0]
    values = sheet_array[:, 1]
    units = sheet_array[:, 2]
    bool_est = sheet_array[:, 3]

    return names, values, units, bool_est

sheet = select_sheet_path()
initial_conditions = read_c0(sheet)
names, values, units, bool_est = read_c0_columns(sheet)
print(bool_est)