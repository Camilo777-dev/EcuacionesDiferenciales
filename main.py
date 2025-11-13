# main_gui.py

import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Importar la lógica del solver
from src.solvers import solve_dynamic_ode

# --- Función auxiliar para nombres ---
def get_deriv_name(order):
    if order == 0: return "y"
    if order == 1: return "y'"
    return f"y{'' * order}"

# --- Clase principal de la Aplicación ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Solucionador de Ecuaciones Diferenciales")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- Variables de estado ---
        self.ic_entry_widgets = [] # Lista para guardar los widgets de C.I.
        self.t_points = np.linspace(0, 10, 500)

        # --- Layout Principal (2 columnas) ---
        self.grid_columnconfigure(0, weight=1) # Columna de inputs
        self.grid_columnconfigure(1, weight=3) # Columna de gráfica
        self.grid_rowconfigure(0, weight=1)

        # --- Frame Izquierdo (Inputs) ---
        self.frame_inputs = ctk.CTkFrame(self, width=350)
        self.frame_inputs.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # --- Frame Derecho (Gráfica y Resultados) ---
        self.frame_outputs = ctk.CTkFrame(self)
        self.frame_outputs.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.frame_outputs.grid_rowconfigure(0, weight=5)
        self.frame_outputs.grid_rowconfigure(1, weight=1)

        # --- Widgets en el Frame Izquierdo ---
        label_title = ctk.CTkLabel(self.frame_inputs, text="Parámetros de la EDO", font=ctk.CTkFont(size=20, weight="bold"))
        label_title.pack(pady=10)

        # 1. Orden de la EDO
        self.order_frame = ctk.CTkFrame(self.frame_inputs, fg_color="transparent")
        self.order_frame.pack(fill="x", padx=10, pady=5)
        
        self.label_order = ctk.CTkLabel(self.order_frame, text="Orden (N):")
        self.label_order.pack(side="left")
        
        self.order_entry = ctk.CTkEntry(self.order_frame, width=50)
        self.order_entry.pack(side="left", padx=5)
        self.order_entry.insert(0, "2")
        
        self.order_button = ctk.CTkButton(self.order_frame, text="Fijar Orden", command=self.update_ic_fields)
        self.order_button.pack(side="left", padx=5)

        # 2. Ecuación
        self.label_nomen = ctk.CTkLabel(self.frame_inputs, text="Nomenclatura: t, y_val, y_p, y_pp", font=ctk.CTkFont(size=10))
        self.label_nomen.pack(pady=(5,0))
        
        self.edo_frame = ctk.CTkFrame(self.frame_inputs, fg_color="transparent")
        self.edo_frame.pack(fill="x", padx=10, pady=5)
        
        self.label_edo = ctk.CTkLabel(self.edo_frame, text="y^(N) =")
        self.label_edo.pack(side="left")
        
        self.edo_entry = ctk.CTkEntry(self.edo_frame, placeholder_text="-2*y_p - y_val")
        self.edo_entry.insert(0, "-2*y_p - y_val")
        self.edo_entry.pack(side="left", fill="x", expand=True, padx=5)

        # 3. Condiciones Iniciales (Dinámico)
        self.ic_frame = ctk.CTkFrame(self.frame_inputs)
        self.ic_frame.pack(fill="x", expand=True, padx=10, pady=10)
        
        # 4. Simulación
        self.sim_frame = ctk.CTkFrame(self.frame_inputs, fg_color="transparent")
        self.sim_frame.pack(fill="x", padx=10, pady=5)
        
        self.label_time = ctk.CTkLabel(self.sim_frame, text="Tiempo Final (t):")
        self.label_time.pack(side="left")
        
        self.time_entry = ctk.CTkEntry(self.sim_frame, width=80)
        self.time_entry.insert(0, "10")
        self.time_entry.pack(side="left", padx=5)

        # 5. Botón de Resolver
        self.solve_button = ctk.CTkButton(self.frame_inputs, text="Resolver y Graficar", command=self.solve_and_plot)
        self.solve_button.pack(fill="x", padx=10, pady=10)
        
        # 6. Etiqueta de Error
        self.error_label = ctk.CTkLabel(self.frame_inputs, text="", text_color="red")
        self.error_label.pack(fill="x", padx=10, pady=5)

        # --- Widgets en el Frame Derecho ---
        
        # 1. La Gráfica
        self.fig, self.ax = plt.subplots(facecolor="#2B2B2B")
        self.ax.set_facecolor("#2B2B2B")
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_outputs)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas.draw()
        
        # 2. Resultado Simbólico
        self.symbolic_textbox = ctk.CTkTextbox(self.frame_outputs, height=100)
        self.symbolic_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.symbolic_textbox.insert("1.0", "Solución Simbólica aparecerá aquí...")
        self.symbolic_textbox.configure(state="disabled")

        # --- Inicializar campos de C.I. ---
        self.update_ic_fields()

    # --- Función para actualizar los campos de C.I. ---
    def update_ic_fields(self):
        try:
            order = int(self.order_entry.get())
            if order < 1:
                self.error_label.configure(text="El orden debe ser >= 1")
                return
            
            # Limpiar widgets anteriores
            for widget in self.ic_frame.winfo_children():
                widget.destroy()
            self.ic_entry_widgets = []
            
            # Actualizar etiqueta de EDO
            self.label_edo.configure(text=f"{get_deriv_name(order)} =")

            # Crear nuevos widgets
            ctk.CTkLabel(self.ic_frame, text="Condiciones Iniciales (t=0):").pack(anchor="w", padx=10)
            
            for i in range(order):
                frame = ctk.CTkFrame(self.ic_frame, fg_color="transparent")
                frame.pack(fill="x", padx=10, pady=2)
                
                label = ctk.CTkLabel(frame, text=f"{get_deriv_name(i)}(0):", width=60)
                label.pack(side="left")
                
                entry = ctk.CTkEntry(frame)
                # Valores default para el ejemplo
                entry.insert(0, "1.0" if i == 0 else "0.0") 
                entry.pack(side="left", fill="x", expand=True)
                
                self.ic_entry_widgets.append(entry)
                
            self.error_label.configure(text="") # Limpiar error
        
        except ValueError:
            self.error_label.configure(text="Error: El orden debe ser un número entero.")
        
    # --- Función para Resolver y Graficar ---
    def solve_and_plot(self):
        try:
            # 1. Recolectar todos los datos
            order = int(self.order_entry.get())
            expr_str = self.edo_entry.get()
            t_end = float(self.time_entry.get())
            
            y0_values = []
            for entry in self.ic_entry_widgets:
                y0_values.append(float(entry.get()))

            if len(y0_values) != order:
                self.error_label.configure(text="Error: Fije el orden antes de resolver.")
                return

            self.t_points = np.linspace(0, t_end, 500)

            # 2. Llamar al solver
            solution_numeric, solution_symbolic_str = solve_dynamic_ode(
                order, expr_str, self.t_points, y0_values
            )

            # 3. Actualizar la gráfica
            self.ax.clear()
            for i in range(order):
                label_name = f"{get_deriv_name(i)}(t)"
                self.ax.plot(self.t_points, solution_numeric[:, i], label=label_name)
            
            self.ax.legend(facecolor="#333333", labelcolor="white")
            self.ax.grid(True, linestyle='--', alpha=0.3)
            self.ax.set_xlabel("Tiempo (t)")
            self.ax.set_ylabel("Valor")
            self.canvas.draw()
            
            # 4. Actualizar el texto simbólico
            self.symbolic_textbox.configure(state="normal")
            self.symbolic_textbox.delete("1.0", "end")
            self.symbolic_textbox.insert("1.0", solution_symbolic_str)
            self.symbolic_textbox.configure(state="disabled")
            
            self.error_label.configure(text="") # Limpiar error
            
        except Exception as e:
            self.error_label.configure(text=f"Error: {e}")

# --- Ejecutar la aplicación ---
if __name__ == "__main__":
    app = App()
    app.mainloop()