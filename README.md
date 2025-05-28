# Telco Churn Project

This project aims to explore and analyze customer churn data from a telecom company. It is organized into modular Python scripts and a dedicated notebook for exploratory data analysis (EDA).

---

## 📁 Project Structure

```
TelcoChurnProject/
├── notebooks/
│   └── eda.ipynb                # EDA notebook
├── src/
│   ├── data/
│   │   └── telco_churn.csv      # Main dataset
│   ├── features/                # Feature engineering code
│   ├── models/                  # Model training and evaluation
│   └── utils/                   # Helper functions
├── requirements.txt             # Project dependencies
├── setup.py                     # Project setup
└── README.md                    # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/TelcoChurnProject.git
cd TelcoChurnProject
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Activate:
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install .
```

> This uses `setup.py` to install the project as a package.

---

## 📊 Running the EDA Notebook

1. Open `notebooks/eda.ipynb` in VS Code or Jupyter Lab.
2. Select the appropriate Python kernel (linked to your virtual environment).
3. Run the cells to perform EDA.

---

## 📦 Requirements

Key libraries used:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `ipykernel` (for running notebooks)

All dependencies are listed in `requirements.txt`.

---

## 🔎 About the Dataset

The main dataset is located at:

```
src/data/telco_churn.csv
```

This dataset should be used for reading and exploration only during EDA.

---

## 🙌 Contributing

Pull requests are welcome! Feel free to suggest improvements, fix bugs, or add new analysis modules.

---