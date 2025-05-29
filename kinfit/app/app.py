import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from pandas import read_csv, DataFrame

class App:

    def __init__(self, tk_root):
        ## Setting window parameters
        self.tk_root = tk_root
        self.tk_root.title("Seletor e Leitor de CSV")
        self.tk_root.geometry("500x300")
        
        ## Attributes
        # File Selector
        self.file_path: str|None = None
        self.file_content: DataFrame|None = None

        # Model Dropdown
        self.model_selected = tk.StringVar()
        self.model_selected.set("Selecione uma opção")

        # Other attributes (Future use)
        
        ## Widgets
        # File selector
        self.select_file_button = tk.Button(
            tk_root,
            text="Selecionar Arquivo CSV",
            command=self._on_selection
        )
        self.select_file_button.pack(pady=10)

        self.file_label = tk.Label(
            tk_root,
            text="Nenhum arquivo selecionado.",
            justify="left"
        )
        self.file_label.pack(pady=10)

        self.data_label = tk.Label(
            tk_root,
            text="Dados do arquivo aparecerão aqui.",
            justify="left"
        )
        self.data_label.pack(pady=10)

        # Select Model Button
        self.model_dropdown = tk.OptionMenu(
            tk_root,
            self.model_selected,
            *["modelo1", "modelo2", "modelo3"]
        )
        self.model_dropdown.pack(pady=20)

        # Run Button
        self.run_button = tk.Button(
            tk_root,
            text="Executar otimização",
            command=self._execute_model
        )


    def _on_selection(self) -> None:
        self._select_file()
        if self.file_path:
            self._read_file(self.file_path)
            self.run_button.pack(pady=10)

    def _select_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Selecione um arquivo",
            filetypes=(("Arquivos CSV", "*.csv"),)
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=self.file_path)
        else:
            messagebox.showinfo("Informação", "Nenhum arquivo selecionado.")

    def _read_file(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as csvfile:
                file_content = read_csv(path)

                self.file_content = file_content
                self.data_label.config(text=self.file_content)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao ler o CSV: {e}")

    def get_file_content(self) -> DataFrame|None:
        return self.file_content


    def _execute_model(self) -> None:
        #Criar uma função para executar o modelo selecionado usando as informações
        # do arquivo selecionado
        messagebox.showinfo("ALERTA", "PARA IMPLEMENTAR")
        
# Execução do app
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()