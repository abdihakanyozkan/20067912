
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MLApp:
   def __init__(self, root):
      self.root = root
      self.root.title("Enhanced ML Application")

      # Notebook for tabs
      self.notebook = ttk.Notebook(root)
      self.notebook.pack(fill='both', expand=True)

      # Create tabs
      self.create_regression_tab()
      self.create_classification_tab()
      self.create_bayesian_tab()

      # Data handling
      self.data = None
      self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

   def create_regression_tab(self):
      # Regression tab with enhanced features
      reg_frame = ttk.Frame(self.notebook)
      self.notebook.add(reg_frame, text="Regression")

      # Model selection
      ttk.Label(reg_frame, text="Model:").grid(row=0, column=0)
      self.reg_model = ttk.Combobox(reg_frame, values=["Linear Regression", "SVR"])
      self.reg_model.grid(row=0, column=1)

      # Loss function selection
      ttk.Label(reg_frame, text="Loss Function:").grid(row=1, column=0)
      self.reg_loss = ttk.Combobox(reg_frame, values=["MSE", "MAE", "Huber Loss"])
      self.reg_loss.grid(row=1, column=1)

      # SVR specific options
      self.svr_kernel_label = ttk.Label(reg_frame, text="Kernel:")
      self.svr_kernel = ttk.Combobox(reg_frame, values=["linear", "rbf", "poly"])

      self.svr_c_label = ttk.Label(reg_frame, text="C:")
      self.svr_c = ttk.Entry(reg_frame)

      self.svr_epsilon_label = ttk.Label(reg_frame, text="Epsilon:")
      self.svr_epsilon = ttk.Entry(reg_frame)

      # Missing data handling
      self.create_missing_data_section(reg_frame, 5)

      # Train button
      ttk.Button(reg_frame, text="Train Model", command=self.train_regression).grid(row=10, column=0, columnspan=2)

      # Visualization area
      self.reg_fig, self.reg_ax = plt.subplots(figsize=(5, 3))
      self.reg_canvas = FigureCanvasTkAgg(self.reg_fig, master=reg_frame)
      self.reg_canvas.get_tk_widget().grid(row=0, column=2, rowspan=11, padx=10)

      # Show/hide SVR options based on model selection
      self.reg_model.bind("<<ComboboxSelected>>", self.toggle_svr_options)

   def create_classification_tab(self):
      # Similar structure to regression tab with classification options
      pass

   def create_bayesian_tab(self):
      # Bayesian tab with GaussianNB options
      pass

   def create_missing_data_section(self, parent, start_row):
      # Missing data handling options
      ttk.Label(parent, text="Missing Data Handling:").grid(row=start_row, column=0)
      self.missing_data_method = ttk.Combobox(parent,
                                              values=["Mean Imputation", "Interpolation", "Forward Fill",
                                                      "Backward Fill"])
      self.missing_data_method.grid(row=start_row, column=1)

   def train_regression(self):
      try:
         if self.data is None:
            messagebox.showerror("Error", "Please load data first!")
            return

         # Handle missing data
         self.handle_missing_data()

         # Split data
         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data.iloc[:, :-1], self.data.iloc[:, -1], test_size=0.2)

         # Train selected model
         model_type = self.reg_model.get()

         if model_type == "Linear Regression":
            model = LinearRegression()
            # Custom loss handling would need custom implementation

         elif model_type == "SVR":
            kernel = self.svr_kernel.get()
            C = float(self.svr_c.get()) if self.svr_c.get() else 1.0
            epsilon = float(self.svr_epsilon.get()) if self.svr_epsilon.get() else 0.1

            model = SVR(kernel=kernel, C=C, epsilon=epsilon)

         model.fit(self.X_train, self.y_train)
         predictions = model.predict(self.X_test)

         # Calculate metrics based on selected loss
         loss_func = self.reg_loss.get()
         if loss_func == "MSE":
            loss = mean_squared_error(self.y_test, predictions)
         elif loss_func == "MAE":
            loss = mean_absolute_error(self.y_test, predictions)
         # Huber loss would need custom implementation

         # Update visualization
         self.update_regression_plot(self.y_test, predictions, loss, loss_func)

      except Exception as e:
         messagebox.showerror("Error", str(e))

   def update_regression_plot(self, y_true, y_pred, loss, loss_func):
      self.reg_ax.clear()
      self.reg_ax.scatter(y_true, y_pred, alpha=0.5)
      self.reg_ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
      self.reg_ax.set_xlabel('True Values')
      self.reg_ax.set_ylabel('Predictions')
      self.reg_ax.set_title(f'Regression Results ({loss_func}: {loss:.2f})')
      self.reg_canvas.draw()

   def handle_missing_data(self):
      method = self.missing_data_method.get()

      if method == "Mean Imputation":
         imputer = SimpleImputer(strategy='mean')
         self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
      elif method == "Interpolation":
         self.data = self.data.interpolate()
      elif method == "Forward Fill":
         self.data = self.data.ffill()
      elif method == "Backward Fill":
         self.data = self.data.bfill()

   def toggle_svr_options(self, event):
      # Show/hide SVR specific options based on model selection
      pass


if __name__ == "__main__":
   root = tk.Tk()
   app = MLApp(root)
   root.mainloop()