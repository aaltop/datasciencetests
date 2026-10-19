import tkinter as tk
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText
from typing import Protocol


class SubmitSearch(Protocol):
    def __call__(self, query: str) -> str: ...

class Search:

    def __init__(self, submit_search: SubmitSearch):

        self.submit_search = submit_search

    def run(self):
        root = tk.Tk()
        root.title("Python Docs")
        root.geometry("640x480")
        root.minsize(640, 480)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        frm = ttk.Frame(root)
        frm.grid(sticky="news")
        frm.grid_rowconfigure(0, weight=0)
        frm.grid_rowconfigure(1, weight=1)
        frm.grid_columnconfigure(0, weight=1)


        # Create search bar
        # -------------------------------------------
        
        search = ttk.Frame(frm)
        search.grid(column=0, row=0, sticky="we", padx=15, pady=10)
        search.grid_columnconfigure(1, weight=1)

        ttk.Label(search, text="Query:").grid(column=0, row=0)
        search_content = tk.StringVar()
        _search_bar = ttk.Entry(search, textvariable=search_content)
        _search_bar.grid(column=1, row=0, sticky="we")
        _button = ttk.Button(search, text="Search")
        _button.grid(column=2, row=0)

        
        # Create search bar
        # ===========================================

        # Create text block
        # -------------------------------------------
        
        output = ttk.Frame(frm)
        output.grid(column=0, row=1, sticky="news")
        output.grid_columnconfigure(0, weight=1)
        output.grid_rowconfigure(0, weight=1)
        _text = ScrolledText(output, width=128)
        _text.grid(row=0, column=0, sticky="ns", padx=15, pady=15)
        _text.insert("end", chars="stuff")
        _text.configure(state="disabled", font=("Helvetica", 12, "normal"))

        # Create text block
        # ===========================================


        def submit_search(*args):
            search_query = search_content.get()
            response = self.submit_search(search_query)
            _text.configure(state="normal")
            _text.replace("0.1", "end", response)
            _text.configure(state="disabled")

        _button.bind("<Return>", submit_search)
        _button.configure(command=submit_search)
        _search_bar.bind("<Return>", submit_search)

        root.mainloop()