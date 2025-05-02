"""
Advanced Machine Learning GUI for Robotics Applications
Student ID: 20067912

Key Features:
1. Comprehensive classical ML algorithms
2. Deep learning with customizable architectures
3. Advanced dimensionality reduction techniques
4. Reinforcement learning environments
5. Specialized tools for robotics applications
"""
import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                             QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                             QDialog, QLineEdit, QTableWidget, QTableWidgetItem,
                             QDialogButtonBox)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, mean_squared_error,
                             confusion_matrix, mean_absolute_error,
                             classification_report)
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
import gym
from collections import deque
import random


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.df = None
        self.target_column = None
        self.layer_config = []
        self.rl_env = None
        self.rl_model = None

        # Create components
        self.create_data_section()
        self.create_missing_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()

            if dataset_name == "Load Custom Dataset":
                return

            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                # Normalize and reshape MNIST data
                self.X_train = X_train.reshape(-1, 28 * 28) / 255.0
                self.X_test = X_test.reshape(-1, 28 * 28) / 255.0
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return

            # Split data
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(data.data, data.target,
                                                 test_size=test_size,
                                                 random_state=42)

            # Apply scaling if selected
            self.apply_scaling()

            self.status_bar.showMessage(f"Loaded {dataset_name}")

        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def load_custom_data(self):
        """Enhanced data loading function with validation and preview"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Open Dataset",
                "",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
            )

            if not file_name:
                return

            if file_name.endswith('.csv'):
                self.df = pd.read_csv(file_name)
            elif file_name.endswith('.xlsx'):
                self.df = pd.read_excel(file_name)
            else:
                raise ValueError("Unsupported file format")

            if self.df.empty:
                raise ValueError("The file is empty")

            if len(self.df.columns) < 2:
                raise ValueError("Dataset needs at least 2 columns")

            row_count, col_count = self.df.shape
            self.status_bar.showMessage(
                f"Loaded: {file_name.split('/')[-1]} | {row_count} rows, {col_count} cols"
            )

            self.select_target_column()
            self.show_data_preview()

        except Exception as e:
            self.show_error(f"Failed to load data: {str(e)}")
            self.df = None

    def show_data_preview(self):
        """Display first 5 rows in a table widget"""
        if not hasattr(self, 'preview_table'):
            self.preview_table = QTableWidget()
            self.preview_table.setAlternatingRowColors(True)
            self.layout.insertWidget(2, self.preview_table)

        df_sample = self.df.head(5)
        rows, cols = df_sample.shape

        self.preview_table.setRowCount(rows)
        self.preview_table.setColumnCount(cols)
        self.preview_table.setHorizontalHeaderLabels(self.df.columns)

        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem(str(df_sample.iloc[i, j]))
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.preview_table.setItem(i, j, item)

        self.preview_table.resizeColumnsToContents()

    def select_target_column(self):
        """Enhanced target column selection"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        label = QLabel("Select the target variable column:")
        layout.addWidget(label)

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Column", "Data Type", "Missing Values"])
        table.setRowCount(len(self.df.columns))

        for i, col in enumerate(self.df.columns):
            table.setItem(i, 0, QTableWidgetItem(col))
            table.setItem(i, 1, QTableWidgetItem(str(self.df[col].dtype)))
            table.setItem(i, 2, QTableWidgetItem(str(self.df[col].isna().sum())))

        table.resizeColumnsToContents()
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(table)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = table.currentRow()
            if selected >= 0:
                target_col = self.df.columns[selected]

                if self.df[target_col].isna().any():
                    self.show_error("Target column contains missing values!")
                    return

                if len(self.df[target_col].unique()) < 2:
                    self.show_error("Target column needs at least 2 unique values")
                    return

                self.target_column = target_col
                self.status_bar.showMessage(f"Selected target: {target_col}")
                self.split_data()

    def split_data(self):
        """Split data into train/test sets"""
        try:
            if self.df is None or self.target_column is None:
                return

            X = self.df.drop(self.target_column, axis=1)
            y = self.df[self.target_column]

            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(X, y, test_size=test_size, random_state=42)

            self.apply_scaling()
            self.status_bar.showMessage("Data split and scaled successfully")

        except Exception as e:
            self.show_error(f"Error splitting data: {str(e)}")

    def apply_scaling(self):
        """Apply selected scaling method"""
        scaling_method = self.scaling_combo.currentText()

        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()

                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)

            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def create_data_section(self):
        """Create the data loading section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])

        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)

        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def create_missing_data_section(self):
        """Create missing data handling section"""
        missing_group = QGroupBox("Missing Data Handling")
        layout = QHBoxLayout()

        self.missing_combo = QComboBox()
        self.missing_combo.addItems([
            "Mean Imputation",
            "Median Imputation",
            "Most Frequent",
            "Drop Missing Values"
        ])

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.handle_missing_data)

        layout.addWidget(QLabel("Strategy:"))
        layout.addWidget(self.missing_combo)
        layout.addWidget(apply_btn)

        missing_group.setLayout(layout)
        self.layout.addWidget(missing_group)

    def handle_missing_data(self):
        """Handle missing data with selected strategy"""
        if self.df is None:
            self.show_error("Load data first!")
            return

        strategy = self.missing_combo.currentText()
        try:
            if strategy == "Drop Missing Values":
                self.df = self.df.dropna()
            else:
                strategy_map = {
                    "Mean Imputation": "mean",
                    "Median Imputation": "median",
                    "Most Frequent": "most_frequent"
                }
                imputer = SimpleImputer(strategy=strategy_map[strategy])
                self.df.iloc[:] = imputer.fit_transform(self.df)

            self.status_bar.showMessage(f"Applied {strategy}")
            self.show_data_preview()

        except Exception as e:
            self.show_error(f"Error: {str(e)}")

    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()

        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]

        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)

        self.layout.addWidget(self.tab_widget)

    def create_classical_ml_tab(self):
        """Create classical ML tab with all algorithms"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()

        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox"}
        )
        regression_layout.addWidget(lr_group)

        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)

        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)

        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()

        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double",
             "priors": ["uniform", "custom"]}
        )
        classification_layout.addWidget(nb_group)

        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int"}
        )
        classification_layout.addWidget(svm_group)

        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)

        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)

        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)

        # SVM for both classification and regression
        svm_dual_group = QGroupBox("SVM (Both Classification/Regression)")
        svm_layout = QVBoxLayout()

        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["linear", "rbf", "poly"])

        self.C_spin = QDoubleSpinBox()
        self.C_spin.setRange(0.1, 10.0)
        self.C_spin.setValue(1.0)

        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1.0)
        self.epsilon_spin.setValue(0.1)

        btn_classify = QPushButton("Train as Classifier")
        btn_classify.clicked.connect(lambda: self.train_svm(is_classification=True))

        btn_regress = QPushButton("Train as Regressor")
        btn_regress.clicked.connect(lambda: self.train_svm(is_classification=False))

        svm_layout.addWidget(QLabel("Kernel:"))
        svm_layout.addWidget(self.kernel_combo)
        svm_layout.addWidget(QLabel("C Parameter:"))
        svm_layout.addWidget(self.C_spin)
        svm_layout.addWidget(QLabel("Epsilon (for SVR):"))
        svm_layout.addWidget(self.epsilon_spin)
        svm_layout.addWidget(btn_classify)
        svm_layout.addWidget(btn_regress)
        svm_dual_group.setLayout(svm_layout)

        layout.addWidget(svm_dual_group, 1, 0)

        return widget

    def train_svm(self, is_classification=True):
        """Train SVM with current parameters"""
        try:
            kernel = self.kernel_combo.currentText()
            C = self.C_spin.value()
            epsilon = self.epsilon_spin.value()

            if is_classification:
                model = SVC(kernel=kernel, C=C, random_state=42)
            else:
                model = SVR(kernel=kernel, C=C, epsilon=epsilon)

            model.fit(self.X_train, self.y_train)
            self.current_model = model
            y_pred = model.predict(self.X_test)

            self.update_visualization(y_pred)
            self.update_metrics(y_pred)
            self.status_bar.showMessage(
                f"SVM {'Classifier' if is_classification else 'Regressor'} trained successfully"
            )

        except Exception as e:
            self.show_error(f"Error training SVM: {str(e)}")

    def create_algorithm_group(self, name, params):
        """Helper to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()

        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))

            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)

            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)

        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)

        group.setLayout(layout)
        return group

    def train_model(self, name, params):
        """Generic model training function"""
        try:
            if name == "Linear Regression":
                model = LinearRegression(
                    fit_intercept=params["fit_intercept"].isChecked(),
                    normalize=params["normalize"].isChecked()
                )
            elif name == "Logistic Regression":
                model = LogisticRegression(
                    C=params["C"].value(),
                    max_iter=params["max_iter"].value(),
                    multi_class=params["multi_class"].currentText(),
                    random_state=42
                )
            elif name == "Naive Bayes":
                model = GaussianNB(
                    var_smoothing=params["var_smoothing"].value()
                )
            elif name == "Support Vector Machine":
                model = SVC(
                    C=params["C"].value(),
                    kernel=params["kernel"].currentText(),
                    degree=params["degree"].value(),
                    random_state=42
                )
            elif name == "Decision Tree":
                model = DecisionTreeClassifier(
                    max_depth=params["max_depth"].value(),
                    min_samples_split=params["min_samples_split"].value(),
                    criterion=params["criterion"].currentText(),
                    random_state=42
                )
            elif name == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=params["n_estimators"].value(),
                    max_depth=params["max_depth"].value(),
                    min_samples_split=params["min_samples_split"].value(),
                    random_state=42
                )
            elif name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(
                    n_neighbors=params["n_neighbors"].value(),
                    weights=params["weights"].currentText(),
                    metric=params["metric"].currentText()
                )
            elif name == "K-Means Parameters":
                model = KMeans(
                    n_clusters=params["n_clusters"].value(),
                    max_iter=params["max_iter"].value(),
                    n_init=params["n_init"].value(),
                    random_state=42
                )
            elif name == "PCA Parameters":
                model = PCA(
                    n_components=params["n_components"].value(),
                    whiten=params["whiten"].isChecked(),
                    random_state=42
                )

            model.fit(self.X_train, self.y_train)
            self.current_model = model
            y_pred = model.predict(self.X_test)

            self.update_visualization(y_pred)
            self.update_metrics(y_pred)
            self.status_bar.showMessage(f"{name} trained successfully")

        except Exception as e:
            self.show_error(f"Error training {name}: {str(e)}")

    def create_deep_learning_tab(self):
        """Create deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()

        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)

        # Layer configuration display
        self.layer_display = QTextEdit()
        self.layer_display.setReadOnly(True)
        mlp_layout.addWidget(self.layer_display)

        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)

        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)

        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)

        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()

        # CNN Architecture
        cnn_arch_group = QGroupBox("CNN Architecture")
        cnn_arch_layout = QVBoxLayout()

        self.cnn_layer_config = []
        cnn_layer_btn = QPushButton("Add CNN Layer")
        cnn_layer_btn.clicked.connect(self.add_cnn_layer_dialog)
        cnn_arch_layout.addWidget(cnn_layer_btn)

        # CNN Layer display
        self.cnn_layer_display = QTextEdit()
        self.cnn_layer_display.setReadOnly(True)
        cnn_arch_layout.addWidget(self.cnn_layer_display)

        cnn_arch_group.setLayout(cnn_arch_layout)
        cnn_layout.addWidget(cnn_arch_group)

        # CNN Training
        cnn_train_btn = QPushButton("Train CNN")
        cnn_train_btn.clicked.connect(self.train_cnn)
        cnn_layout.addWidget(cnn_train_btn)

        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)

        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()

        # RNN Architecture
        rnn_arch_group = QGroupBox("RNN Architecture")
        rnn_arch_layout = QVBoxLayout()

        self.rnn_layer_config = []
        rnn_layer_btn = QPushButton("Add RNN Layer")
        rnn_layer_btn.clicked.connect(self.add_rnn_layer_dialog)
        rnn_arch_layout.addWidget(rnn_layer_btn)

        # RNN Layer display
        self.rnn_layer_display = QTextEdit()
        self.rnn_layer_display.setReadOnly(True)
        rnn_arch_layout.addWidget(self.rnn_layer_display)

        rnn_arch_group.setLayout(rnn_arch_layout)
        rnn_layout.addWidget(rnn_arch_group)

        # RNN Training
        rnn_train_btn = QPushButton("Train RNN")
        rnn_train_btn.clicked.connect(self.train_rnn)
        rnn_layout.addWidget(rnn_train_btn)

        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)

        return widget

    def add_layer_dialog(self):
        """Dialog to add neural network layers"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)

        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()

        self.layer_param_inputs = {}

        def update_params():
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()

            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input

                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh"])
                self.layer_param_inputs["activation"] = activation_combo

                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input

                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

        type_combo.currentIndexChanged.connect(update_params)
        update_params()

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def add_layer():
            layer_type = type_combo.currentText()
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))

            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            self.update_layer_display()
            dialog.accept()

        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def update_layer_display(self):
        """Update the layer configuration display"""
        display_text = "Current Network Architecture:\n\n"
        for i, layer in enumerate(self.layer_config, 1):
            display_text += f"Layer {i}: {layer['type']}\n"
            for param, value in layer['params'].items():
                display_text += f"  {param}: {value}\n"
            display_text += "\n"

        self.layer_display.setPlainText(display_text)

    def add_cnn_layer_dialog(self):
        """Dialog to add CNN layers"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add CNN Layer")
        layout = QVBoxLayout(dialog)

        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()

        self.cnn_layer_param_inputs = {}

        def update_params():
            for widget in list(self.cnn_layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.cnn_layer_param_inputs.clear()

            layer_type = type_combo.currentText()
            if layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.cnn_layer_param_inputs["filters"] = filters_input

                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.cnn_layer_param_inputs["kernel_size"] = kernel_input

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh"])
                self.cnn_layer_param_inputs["activation"] = activation_combo

                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "MaxPooling2D":
                pool_label = QLabel("Pool Size:")
                pool_input = QLineEdit()
                pool_input.setText("2, 2")
                self.cnn_layer_param_inputs["pool_size"] = pool_input

                params_layout.addWidget(pool_label)
                params_layout.addWidget(pool_input)

            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.cnn_layer_param_inputs["rate"] = rate_input

                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

        type_combo.currentIndexChanged.connect(update_params)
        update_params()

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def add_layer():
            layer_type = type_combo.currentText()
            layer_params = {}
            for param_name, widget in self.cnn_layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    if param_name in ["kernel_size", "pool_size"]:
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))

            self.cnn_layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            self.update_cnn_layer_display()
            dialog.accept()

        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def update_cnn_layer_display(self):
        """Update the CNN layer configuration display"""
        display_text = "Current CNN Architecture:\n\n"
        for i, layer in enumerate(self.cnn_layer_config, 1):
            display_text += f"Layer {i}: {layer['type']}\n"
            for param, value in layer['params'].items():
                display_text += f"  {param}: {value}\n"
            display_text += "\n"

        self.cnn_layer_display.setPlainText(display_text)

    def add_rnn_layer_dialog(self):
        """Dialog to add RNN layers"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add RNN Layer")
        layout = QVBoxLayout(dialog)

        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["SimpleRNN", "LSTM", "GRU", "Dense", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()

        self.rnn_layer_param_inputs = {}

        def update_params():
            for widget in list(self.rnn_layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.rnn_layer_param_inputs.clear()

            layer_type = type_combo.currentText()
            if layer_type in ["SimpleRNN", "LSTM", "GRU"]:
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.rnn_layer_param_inputs["units"] = units_input

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["tanh", "relu", "sigmoid"])
                self.rnn_layer_param_inputs["activation"] = activation_combo

                return_sequences_label = QLabel("Return Sequences:")
                return_sequences_check = QCheckBox()
                self.rnn_layer_param_inputs["return_sequences"] = return_sequences_check

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
                params_layout.addWidget(return_sequences_label)
                params_layout.addWidget(return_sequences_check)

            elif layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.rnn_layer_param_inputs["units"] = units_input

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.rnn_layer_param_inputs["activation"] = activation_combo

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.rnn_layer_param_inputs["rate"] = rate_input

                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

        type_combo.currentIndexChanged.connect(update_params)
        update_params()

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def add_layer():
            layer_type = type_combo.currentText()
            layer_params = {}
            for param_name, widget in self.rnn_layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    layer_params[param_name] = widget.isChecked()

            self.rnn_layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            self.update_rnn_layer_display()
            dialog.accept()

        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def update_rnn_layer_display(self):
        """Update the RNN layer configuration display"""
        display_text = "Current RNN Architecture:\n\n"
        for i, layer in enumerate(self.rnn_layer_config, 1):
            display_text += f"Layer {i}: {layer['type']}\n"
            for param, value in layer['params'].items():
                display_text += f"  {param}: {value}\n"
            display_text += "\n"

        self.rnn_layer_display.setPlainText(display_text)

    def create_training_params_group(self):
        """Create training parameters group"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()

        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)

        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)

        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)

        # Add optimizer selection
        optimizer_layout = QHBoxLayout()
        optimizer_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "rmsprop"])
        optimizer_layout.addWidget(self.optimizer_combo)
        layout.addLayout(optimizer_layout)

        # Add loss function selection
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Loss Function:"))
        self.loss_combo = QComboBox()
        self.loss_combo.addItems([
            "categorical_crossentropy",
            "binary_crossentropy",
            "mse",
            "mae"
        ])
        loss_layout.addWidget(self.loss_combo)
        layout.addLayout(loss_layout)

        group.setLayout(layout)
        return group

    def train_neural_network(self):
        """Train the neural network"""
        if not self.layer_config:
            self.show_error("Please add at least one layer")
            return

        try:
            model = models.Sequential()
            for layer_config in self.layer_config:
                layer_type = layer_config["type"]
                params = layer_config["params"]

                if layer_type == "Dense":
                    model.add(layers.Dense(**params))
                elif layer_type == "Conv2D":
                    if len(model.layers) == 0:
                        params['input_shape'] = self.X_train.shape[1:]
                    model.add(layers.Conv2D(**params))
                elif layer_type == "MaxPooling2D":
                    model.add(layers.MaxPooling2D())
                elif layer_type == "Flatten":
                    model.add(layers.Flatten())
                elif layer_type == "Dropout":
                    model.add(layers.Dropout(**params))

            # Add output layer
            if len(np.unique(self.y_train)) > 2:  # Multiclass classification
                num_classes = len(np.unique(self.y_train))
                model.add(layers.Dense(num_classes, activation='softmax'))
            else:  # Binary classification or regression
                if self.loss_combo.currentText() in ["mse", "mae"]:
                    model.add(layers.Dense(1))
                else:
                    model.add(layers.Dense(1, activation='sigmoid'))

            # Select optimizer
            optimizer_name = self.optimizer_combo.currentText()
            if optimizer_name == "adam":
                optimizer = optimizers.Adam(learning_rate=self.lr_spin.value())
            elif optimizer_name == "sgd":
                optimizer = optimizers.SGD(learning_rate=self.lr_spin.value())
            elif optimizer_name == "rmsprop":
                optimizer = optimizers.RMSprop(learning_rate=self.lr_spin.value())

            # Compile model
            loss_fn = self.loss_combo.currentText()
            metrics = ['accuracy'] if loss_fn in ["categorical_crossentropy", "binary_crossentropy"] else ['mae', 'mse']

            model.compile(optimizer=optimizer,
                          loss=loss_fn,
                          metrics=metrics)

            # Prepare data
            if len(self.y_train.shape) == 1 or self.y_train.shape[1] == 1:
                if loss_fn in ["categorical_crossentropy"]:
                    y_train = tf.keras.utils.to_categorical(self.y_train)
                    y_test = tf.keras.utils.to_categorical(self.y_test)
                else:
                    y_train = self.y_train
                    y_test = self.y_test
            else:
                y_train = self.y_train
                y_test = self.y_test

            # Train model
            history = model.fit(
                self.X_train, y_train,
                batch_size=self.batch_size_spin.value(),
                epochs=self.epochs_spin.value(),
                validation_data=(self.X_test, y_test),
                callbacks=[self.create_progress_callback()],
                verbose=0
            )

            self.current_model = model
            self.plot_training_history(history)
            self.status_bar.showMessage("Neural Network Training Complete")

            # Evaluate model
            y_pred = model.predict(self.X_test)
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
                y_test = np.argmax(y_test, axis=1)
            else:
                y_pred = y_pred.flatten()

            self.update_metrics(y_pred, y_test)

        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")

    def train_cnn(self):
        """Train the CNN model"""
        if not self.cnn_layer_config:
            self.show_error("Please add at least one CNN layer")
            return

        try:
            model = models.Sequential()

            # Add CNN layers
            for layer_config in self.cnn_layer_config:
                layer_type = layer_config["type"]
                params = layer_config["params"]

                if layer_type == "Conv2D":
                    if len(model.layers) == 0:
                        params['input_shape'] = self.X_train.shape[1:]
                    model.add(layers.Conv2D(**params))
                elif layer_type == "MaxPooling2D":
                    model.add(layers.MaxPooling2D(**params))
                elif layer_type == "Flatten":
                    model.add(layers.Flatten())
                elif layer_type == "Dropout":
                    model.add(layers.Dropout(**params))

            # Add output layer
            if len(np.unique(self.y_train)) > 2:  # Multiclass classification
                num_classes = len(np.unique(self.y_train))
                model.add(layers.Dense(num_classes, activation='softmax'))
            else:  # Binary classification or regression
                if self.loss_combo.currentText() in ["mse", "mae"]:
                    model.add(layers.Dense(1))
                else:
                    model.add(layers.Dense(1, activation='sigmoid'))

            # Select optimizer
            optimizer_name = self.optimizer_combo.currentText()
            if optimizer_name == "adam":
                optimizer = optimizers.Adam(learning_rate=self.lr_spin.value())
            elif optimizer_name == "sgd":
                optimizer = optimizers.SGD(learning_rate=self.lr_spin.value())
            elif optimizer_name == "rmsprop":
                optimizer = optimizers.RMSprop(learning_rate=self.lr_spin.value())

            # Compile model
            loss_fn = self.loss_combo.currentText()
            metrics = ['accuracy'] if loss_fn in ["categorical_crossentropy", "binary_crossentropy"] else ['mae', 'mse']

            model.compile(optimizer=optimizer,
                          loss=loss_fn,
                          metrics=metrics)

            # Prepare data
            if len(self.y_train.shape) == 1 or self.y_train.shape[1] == 1:
                if loss_fn in ["categorical_crossentropy"]:
                    y_train = tf.keras.utils.to_categorical(self.y_train)
                    y_test = tf.keras.utils.to_categorical(self.y_test)
                else:
                    y_train = self.y_train
                    y_test = self.y_test
            else:
                y_train = self.y_train
                y_test = self.y_test

            # Train model
            history = model.fit(
                self.X_train, y_train,
                batch_size=self.batch_size_spin.value(),
                epochs=self.epochs_spin.value(),
                validation_data=(self.X_test, y_test),
                callbacks=[self.create_progress_callback()],
                verbose=0
            )

            self.current_model = model
            self.plot_training_history(history)
            self.status_bar.showMessage("CNN Training Complete")

            # Evaluate model
            y_pred = model.predict(self.X_test)
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
                y_test = np.argmax(y_test, axis=1)
            else:
                y_pred = y_pred.flatten()

            self.update_metrics(y_pred, y_test)

        except Exception as e:
            self.show_error(f"Error training CNN: {str(e)}")

    def train_rnn(self):
        """Train the RNN model"""
        if not self.rnn_layer_config:
            self.show_error("Please add at least one RNN layer")
            return

        try:
            model = models.Sequential()

            # Add RNN layers
            for i, layer_config in enumerate(self.rnn_layer_config):
                layer_type = layer_config["type"]
                params = layer_config["params"]

                if layer_type == "SimpleRNN":
                    if i == 0:  # First layer needs input shape
                        params['input_shape'] = (self.X_train.shape[1], 1)
                    model.add(layers.SimpleRNN(**params))
                elif layer_type == "LSTM":
                    if i == 0:  # First layer needs input shape
                        params['input_shape'] = (self.X_train.shape[1], 1)
                    model.add(layers.LSTM(**params))
                elif layer_type == "GRU":
                    if i == 0:  # First layer needs input shape
                        params['input_shape'] = (self.X_train.shape[1], 1)
                    model.add(layers.GRU(**params))
                elif layer_type == "Dense":
                    model.add(layers.Dense(**params))
                elif layer_type == "Dropout":
                    model.add(layers.Dropout(**params))

            # Add output layer
            if len(np.unique(self.y_train)) > 2:  # Multiclass classification
                num_classes = len(np.unique(self.y_train))
                model.add(layers.Dense(num_classes, activation='softmax'))
            else:  # Binary classification or regression
                if self.loss_combo.currentText() in ["mse", "mae"]:
                    model.add(layers.Dense(1))
                else:
                    model.add(layers.Dense(1, activation='sigmoid'))

            # Select optimizer
            optimizer_name = self.optimizer_combo.currentText()
            if optimizer_name == "adam":
                optimizer = optimizers.Adam(learning_rate=self.lr_spin.value())
            elif optimizer_name == "sgd":
                optimizer = optimizers.SGD(learning_rate=self.lr_spin.value())
            elif optimizer_name == "rmsprop":
                optimizer = optimizers.RMSprop(learning_rate=self.lr_spin.value())

            # Compile model
            loss_fn = self.loss_combo.currentText()
            metrics = ['accuracy'] if loss_fn in ["categorical_crossentropy", "binary_crossentropy"] else ['mae', 'mse']

            model.compile(optimizer=optimizer,
                          loss=loss_fn,
                          metrics=metrics)

            # Prepare data
            if len(self.y_train.shape) == 1 or self.y_train.shape[1] == 1:
                if loss_fn in ["categorical_crossentropy"]:
                    y_train = tf.keras.utils.to_categorical(self.y_train)
                    y_test = tf.keras.utils.to_categorical(self.y_test)
                else:
                    y_train = self.y_train
                    y_test = self.y_test
            else:
                y_train = self.y_train
                y_test = self.y_test

            # Reshape data for RNN (samples, timesteps, features)
            X_train_rnn = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
            X_test_rnn = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

            # Train model
            history = model.fit(
                X_train_rnn, y_train,
                batch_size=self.batch_size_spin.value(),
                epochs=self.epochs_spin.value(),
                validation_data=(X_test_rnn, y_test),
                callbacks=[self.create_progress_callback()],
                verbose=0
            )

            self.current_model = model
            self.plot_training_history(history)
            self.status_bar.showMessage("RNN Training Complete")

            # Evaluate model
            y_pred = model.predict(X_test_rnn)
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
                y_test = np.argmax(y_test, axis=1)
            else:
                y_pred = y_pred.flatten()

            self.update_metrics(y_pred, y_test)

        except Exception as e:
            self.show_error(f"Error training RNN: {str(e)}")

    def plot_training_history(self, history):
        """Plot training history"""
        self.figure.clear()

        # Plot loss
        ax1 = self.figure.add_subplot(121)
        ax1.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()

        # Plot accuracy or other metric
        ax2 = self.figure.add_subplot(122)
        metric_name = [m for m in history.history.keys() if m not in ['loss', 'val_loss']][0]
        ax2.plot(history.history[metric_name], label=f'Training {metric_name}')
        if f'val_{metric_name}' in history.history:
            ax2.plot(history.history[f'val_{metric_name}'], label=f'Validation {metric_name}')
        ax2.set_title(f'Model {metric_name}')
        ax2.set_ylabel(metric_name)
        ax2.set_xlabel('Epoch')
        ax2.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def run_dim_reduction(self):
        try:
            if self.X_train is None or self.X_test is None:
                self.show_error("Please load and split data first.")
                return

            algo = self.dim_algo_combo.currentText()
            n_components = self.n_components_spin.value()

            if algo == "LDA":
                if self.y_train is None:
                    self.show_error("LDA requires target labels.")
                    return
                model = LDA(n_components=n_components)
                reduced = model.fit_transform(self.X_train, self.y_train)

            elif algo == "t-SNE":
                perplexity = self.perplexity_spin.value()
                model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
                reduced = model.fit_transform(self.X_train)

            elif algo == "UMAP":
                model = umap.UMAP(n_components=n_components, random_state=42)
                reduced = model.fit_transform(self.X_train)

            # Görselleştir
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=self.y_train, cmap='Spectral')
            self.figure.colorbar(scatter)
            ax.set_title(f"{algo} Projection")
            self.canvas.draw()

        except Exception as e:
            self.show_error(f"Error in dimensionality reduction: {str(e)}")

    def create_progress_callback(self):
        """Create training progress callback"""

        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar

            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)

        return ProgressCallback(self.progress_bar)

    def create_dim_reduction_tab(self):
        """Create dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()

        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)

        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        # LDA / t-SNE / UMAP Group
        other_group = QGroupBox("LDA / t-SNE / UMAP")
        other_layout = QVBoxLayout()

        self.dim_algo_combo = QComboBox()
        self.dim_algo_combo.addItems(["LDA", "t-SNE", "UMAP"])
        other_layout.addWidget(QLabel("Select Reduction Method:"))
        other_layout.addWidget(self.dim_algo_combo)

        self.perplexity_spin = QSpinBox()
        self.perplexity_spin.setRange(5, 100)
        self.perplexity_spin.setValue(30)
        other_layout.addWidget(QLabel("Perplexity (for t-SNE):"))
        other_layout.addWidget(self.perplexity_spin)

        self.n_components_spin = QSpinBox()
        self.n_components_spin.setRange(1, 3)
        self.n_components_spin.setValue(2)
        other_layout.addWidget(QLabel("Components:"))
        other_layout.addWidget(self.n_components_spin)

        run_btn = QPushButton("Apply Dimensionality Reduction")
        run_btn.clicked.connect(self.run_dim_reduction)
        other_layout.addWidget(run_btn)

        other_group.setLayout(other_layout)
        layout.addWidget(other_group, 1, 0, 1, 2)

        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()

        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)

        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)

        return widget

    def create_rl_tab(self):
        """Create reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()

        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)

        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)

        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()

        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)

        # RL Parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()

        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 10000)
        self.episodes_spin.setValue(100)
        params_layout.addWidget(QLabel("Episodes:"))
        params_layout.addWidget(self.episodes_spin)

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 1.0)
        self.gamma_spin.setValue(0.99)
        params_layout.addWidget(QLabel("Discount Factor (gamma):"))
        params_layout.addWidget(self.gamma_spin)

        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0, 1.0)
        self.epsilon_spin.setValue(0.1)
        params_layout.addWidget(QLabel("Exploration Rate (epsilon):"))
        params_layout.addWidget(self.epsilon_spin)

        self.lr_spin_rl = QDoubleSpinBox()
        self.lr_spin_rl.setRange(0.0, 1.0)
        self.lr_spin_rl.setValue(0.001)
        params_layout.addWidget(QLabel("Learning Rate:"))
        params_layout.addWidget(self.lr_spin_rl)

        params_group.setLayout(params_layout)
        algo_layout.addWidget(params_group)

        # Training controls
        train_btn = QPushButton("Train RL Agent")
        train_btn.clicked.connect(self.train_rl_agent)
        algo_layout.addWidget(train_btn)

        # Visualization button
        viz_btn = QPushButton("Show Training Progress")
        viz_btn.clicked.connect(self.show_rl_progress)
        algo_layout.addWidget(viz_btn)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)

        return widget

    def train_rl_agent(self):
        """Train the RL agent"""
        try:
            env_name = self.env_combo.currentText()
            algo_name = self.rl_algo_combo.currentText()

            # Create environment
            self.rl_env = gym.make(env_name)
            state_size = self.rl_env.observation_space.shape[0]
            action_size = self.rl_env.action_space.n

            # Get parameters
            episodes = self.episodes_spin.value()
            gamma = self.gamma_spin.value()
            epsilon = self.epsilon_spin.value()
            learning_rate = self.lr_spin_rl.value()

            # Initialize algorithm
            if algo_name == "Q-Learning":
                self.rl_model = QLearningAgent(
                    state_size,
                    action_size,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    epsilon=epsilon
                )
            elif algo_name == "SARSA":
                self.rl_model = SARSAAgent(
                    state_size,
                    action_size,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    epsilon=epsilon
                )
            elif algo_name == "DQN":
                self.rl_model = DQNAgent(
                    state_size,
                    action_size,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    epsilon=epsilon
                )

            # Train the agent
            self.rl_rewards = []
            for e in range(episodes):
                state = self.rl_env.reset()
                state = np.reshape(state, [1, state_size])
                total_reward = 0
                done = False

                while not done:
                    action = self.rl_model.act(state)
                    next_state, reward, done, _ = self.rl_env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])

                    if algo_name == "Q-Learning":
                        self.rl_model.learn(state, action, reward, next_state, done)
                    elif algo_name == "SARSA":
                        next_action = self.rl_model.act(next_state)
                        self.rl_model.learn(state, action, reward, next_state, next_action, done)
                    elif algo_name == "DQN":
                        self.rl_model.remember(state, action, reward, next_state, done)
                        self.rl_model.replay(self.rl_model.batch_size)

                    state = next_state
                    total_reward += reward

                self.rl_rewards.append(total_reward)
                self.progress_bar.setValue(int((e + 1) / episodes * 100))

                # Print progress
                if (e + 1) % 10 == 0:
                    self.status_bar.showMessage(
                        f"Episode: {e + 1}/{episodes}, Reward: {total_reward}"
                    )

            self.status_bar.showMessage(
                f"Training complete! Average reward: {np.mean(self.rl_rewards[-10:])}"
            )

        except Exception as e:
            self.show_error(f"Error in RL training: {str(e)}")

    def show_rl_progress(self):
        """Show RL training progress"""
        if not hasattr(self, 'rl_rewards') or not self.rl_rewards:
            self.show_error("No RL training data available")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.rl_rewards)
        ax.set_title("RL Training Progress")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        self.canvas.draw()

    def create_visualization(self):
        """Create visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)

        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

    def update_visualization(self, y_pred, y_true=None):
        """Update visualization with results"""
        if y_true is None:
            y_true = self.y_test

        self.figure.clear()

        if len(np.unique(y_true)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(y_true, y_pred)
            ax.plot([y_true.min(), y_true.max()],
                    [y_true.min(), y_true.max()],
                    'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Regression Results")
        else:  # Classification
            if self.X_train.shape[1] > 2:
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                     c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                ax.set_title("Classification Results (PCA Projection)")
            else:
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                     c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                ax.set_title("Classification Results")

        self.canvas.draw()

    def update_metrics(self, y_pred, y_true=None):
        """Update metrics display"""
        if y_true is None:
            y_true = self.y_test

        if len(np.unique(y_true)) > 10:  # Regression
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = 1 - (mse / np.var(y_true))

            metrics_text = f"""
            Regression Metrics:
            -------------------
            Mean Squared Error: {mse:.4f}
            Mean Absolute Error: {mae:.4f}
            R-squared: {r2:.4f}
            """
        else:  # Classification
            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred)

            metrics_text = f"""
            Classification Metrics:
            ----------------------
            Accuracy: {accuracy:.4f}

            Confusion Matrix:
            {cm}

            Classification Report:
            {report}
            """

        self.metrics_text.setPlainText(metrics_text)

    def create_status_bar(self):
        """Create status bar with progress bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)


class QLearningAgent:
    """Q-Learning agent implementation"""

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.amax(self.q_table[next_state]) * (not done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])


class SARSAAgent:
    """SARSA agent implementation"""

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, next_action, done):
        target = reward + self.gamma * self.q_table[next_state][next_action] * (not done)
        self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])


class DQNAgent:
    """Deep Q-Network agent implementation"""

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.batch_size = 32

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def create_tabs(self):
        """Create tabs for different ML topics including new advanced tab"""
        self.tab_widget = QTabWidget()

        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Advanced Features", self.create_advanced_tab)  # New tab added here
        ]

        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)

        self.layout.addWidget(self.tab_widget)

    def create_advanced_tab(self):
        """Create the new advanced features tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Add all the new advanced components here
        self.create_dim_reduction_advanced_group(layout)
        self.create_clustering_advanced_group(layout)
        self.create_validation_advanced_group(layout)
        self.create_eigenvector_group(layout)

        return widget

    def create_dim_reduction_advanced_group(self, parent_layout):
        """Advanced dimensionality reduction components"""
        group = QGroupBox("Advanced Dimensionality Reduction")
        layout = QGridLayout()

        # PCA with Explained Variance
        pca_group = QGroupBox("PCA with Explained Variance")
        pca_layout = QVBoxLayout()

        self.pca_var_slider = QSlider(Qt.Orientation.Horizontal)
        self.pca_var_slider.setRange(1, 100)
        self.pca_var_slider.setValue(95)

        self.pca_var_label = QLabel("Variance Threshold: 95%")
        self.pca_var_slider.valueChanged.connect(
            lambda: self.pca_var_label.setText(f"Variance Threshold: {self.pca_var_slider.value()}%")
        )

        pca_btn = QPushButton("Run PCA with Variance Threshold")
        pca_btn.clicked.connect(self.run_pca_with_variance)

        pca_layout.addWidget(self.pca_var_label)
        pca_layout.addWidget(self.pca_var_slider)
        pca_layout.addWidget(pca_btn)
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 0)

        # [Rest of the advanced dimensionality reduction components...]

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_clustering_advanced_group(self, parent_layout):
        """Advanced clustering components"""
        group = QGroupBox("Advanced Clustering Techniques")
        layout = QGridLayout()

        # K-Means with Elbow Method
        elbow_group = QGroupBox("K-Means with Elbow Method")
        elbow_layout = QVBoxLayout()

        self.max_clusters_spin = QSpinBox()
        self.max_clusters_spin.setRange(2, 20)
        self.max_clusters_spin.setValue(10)

        elbow_btn = QPushButton("Find Optimal Clusters (Elbow Method)")
        elbow_btn.clicked.connect(self.run_elbow_method)

        elbow_layout.addWidget(QLabel("Max Clusters:"))
        elbow_layout.addWidget(self.max_clusters_spin)
        elbow_layout.addWidget(elbow_btn)
        elbow_group.setLayout(elbow_layout)
        layout.addWidget(elbow_group, 0, 0)

        # [Rest of the advanced clustering components...]

        group.setLayout(layout)
        parent_layout.addWidget(group)

    # [All the new advanced methods...]
    def run_pca_with_variance(self):
        """Run PCA with explained variance threshold"""
        if self.X_train is None:
            self.show_error("Please load data first!")
            return

        try:
            var_threshold = self.pca_var_slider.value() / 100
            pca = PCA(n_components=var_threshold)
            pca.fit(self.X_train)

            # Plot explained variance
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            cumsum = np.cumsum(pca.explained_variance_ratio_)
            ax.plot(range(1, len(cumsum) + 1), cumsum, 'b-o')
            ax.axhline(y=var_threshold, color='r', linestyle='--')
            ax.axvline(x=np.where(cumsum >= var_threshold)[0][0] + 1, color='g', linestyle='--')

            ax.set_title(f"Explained Variance (Threshold: {var_threshold * 100:.0f}%)")
            ax.set_xlabel("Number of Components")
            ax.set_ylabel("Cumulative Explained Variance")
            ax.grid(True)

            self.canvas.draw()

            # Project data
            X_pca = pca.transform(self.X_train)
            self.status_bar.showMessage(
                f"PCA reduced data from {self.X_train.shape[1]} to {X_pca.shape[1]} dimensions "
                f"(explaining {cumsum[-1] * 100:.1f}% variance)"
            )

        except Exception as e:
            self.show_error(f"Error in PCA: {str(e)}")

    def run_elbow_method(self):
        """Run elbow method to find optimal number of clusters"""
        if self.X_train is None:
            self.show_error("Please load data first!")
            return

        try:
            max_clusters = self.max_clusters_spin.value()
            distortions = []

            for k in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(self.X_train)
                distortions.append(kmeans.inertia_)

            # Plot elbow curve
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            ax.plot(range(1, max_clusters + 1), distortions, 'b-o')
            ax.set_title('Elbow Method for Optimal k')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('Distortion')
            ax.grid(True)

            self.canvas.draw()
            self.status_bar.showMessage(
                "Elbow method completed. Look for the 'elbow' point in the plot."
            )

        except Exception as e:
            self.show_error(f"Error in elbow method: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())