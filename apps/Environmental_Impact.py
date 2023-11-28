import tkinter as tk
from tkinter import ttk
# Assuming you have the chemicals library installed
import chemicals.environment

def retrieve_data():
    CASRN_value = CASRN_entry.get()

    GWP_methods = chemicals.environment.GWP_methods(CASRN_value)
    GWP_values = {method: chemicals.environment.GWP(CASRN_value, method=method) for method in GWP_methods}

    ODP_methods = chemicals.environment.ODP_methods(CASRN_value)
    ODP_values = {method: chemicals.environment.ODP(CASRN_value, method=method) for method in ODP_methods}

    logP_methods = chemicals.environment.logP_methods(CASRN_value)
    logP_values = {method: chemicals.environment.logP(CASRN_value, method=method) for method in logP_methods}

    result_text.set("Results for CASRN {}: \n\n".format(CASRN_value))
    result_text.set(result_text.get() + "GWP values:\n")
    for method, value in GWP_values.items():
        result_text.set(result_text.get() + "{}: {}\n".format(method, value))
    
    result_text.set(result_text.get() + "\nODP values:\n")
    for method, value in ODP_values.items():
        result_text.set(result_text.get() + "{}: {}\n".format(method, value))
    
    result_text.set(result_text.get() + "\nlogP values:\n")
    for method, value in logP_values.items():
        result_text.set(result_text.get() + "{}: {}\n".format(method, value))

app = tk.Tk()
app.title("Environmental Properties Retriever")

frame = ttk.Frame(app, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

CASRN_label = ttk.Label(frame, text="Enter CASRN:")
CASRN_label.grid(row=0, column=0, sticky=tk.W, pady=5)

CASRN_entry = ttk.Entry(frame, width=20)
CASRN_entry.grid(row=0, column=1, sticky=tk.W, pady=5)

submit_button = ttk.Button(frame, text="Retrieve Data", command=retrieve_data)
submit_button.grid(row=1, column=0, columnspan=2, pady=10)

result_label = ttk.Label(frame, text="Results:")
result_label.grid(row=2, column=0, sticky=tk.W, pady=5)

result_text = tk.StringVar()
result_display = ttk.Label(frame, textvariable=result_text)
result_display.grid(row=3, column=0, columnspan=2, sticky=tk.W)

app.mainloop()
