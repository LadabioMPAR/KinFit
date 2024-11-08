import pandas as pd
import tkinter as tk
from tkinter import filedialog

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

