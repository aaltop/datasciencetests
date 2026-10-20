from pathlib import Path
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText
from typing import TypedDict
import abc

class TextLocation(TypedDict):
    file: str
    start_row: int
    end_row: int
    text: str


class Search(abc.ABC):

    def __init__(self):

        self.text_locations: list[TextLocation] = []

    @abc.abstractmethod
    def submit_search(self, query: str) -> list[TextLocation]:
        """
        Query a set of files for certain content.
        """

    @abc.abstractmethod
    def get_file(self, file_name: str) -> str:
        """
        Get the file contents of the passed file name.
        """

    def run(self):
        root = tk.Tk()
        root.title("Python Docs")
        root.geometry("640x480")
        root.minsize(640, 480)
        root.grid_columnconfigure([0,2], weight=1, minsize=64)
        root.grid_columnconfigure(1, weight=2)
        root.grid_rowconfigure(0, weight=1)

        left_frm = ttk.Frame(root)
        left_frm.grid(column=0, row=0, sticky="news")
        left_frm.grid_rowconfigure(0, weight=1)
        left_frm.grid_columnconfigure(0, weight=1)
        files_list = tk.Listbox(left_frm)
        files_list.grid(column=0, row=0, sticky="news", padx=10, pady=10)

        frm = ttk.Frame(root)
        frm.grid(column=1, row=0, sticky="news")
        frm.grid_rowconfigure(0, weight=0)
        frm.grid_rowconfigure(1, weight=1)
        frm.grid_columnconfigure(0, weight=1)

        right_frm = ttk.Frame(root)
        right_frm.grid(column=2, row=0, sticky="news")

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
        _text = ScrolledText(output)
        _text.grid(row=0, column=0, sticky="ns", padx=15, pady=15)
        _text.insert("end", chars="stuff")
        _text.configure(state="disabled", font=("Helvetica", 12, "normal"))

        # Create text block
        # ===========================================

        def insert_text(text: str):

            _text.configure(state="normal")
            _text.replace("0.1", "end", text)
            _text.configure(state="disabled")


        def insert_file(filename: str):
            insert_text(self.get_file(filename))

        def insert_highlight_text(info: TextLocation):
            insert_file(info["file"])
            tagname = "highlight"
            _text.tag_delete(tagname)
            start_index = f"{info['start_row']}.0"
            _text.tag_add(tagname, start_index, f"{info["end_row"]}.end")
            _text.tag_configure(tagname, background="yellow")
            _text.see(start_index)

        def submit_search(*args):
            search_query = search_content.get()
            response = self.submit_search(search_query)
            files_list.delete(0, "end")
            self.text_locations = response
            files_list.insert("end", *(content["file"] for content in response))
            if len(response) > 0:
                insert_highlight_text(response[0])
                files_list.activate(0)

        _button.bind("<Return>", submit_search)
        _button.configure(command=submit_search)
        _search_bar.bind("<Return>", submit_search)

        def select_file(*args):
            selections = files_list.curselection()
            if len(selections) < 1:
                return
            selected_idx: int = selections[0]
            if selected_idx < len(self.text_locations):
                insert_highlight_text(self.text_locations[selected_idx])

        files_list.bind("<<ListboxSelect>>", select_file)

        root.mainloop()