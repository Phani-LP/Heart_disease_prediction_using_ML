# Heart Disease Prediction Using Machine Learning

This project is a web-based application for predicting heart disease using machine learning algorithms. Built with Python, Django, and various machine learning techniques, it provides an intuitive interface for users to input medical data and receive predictions about the likelihood of heart disease.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Files and Directories](#files-and-directories)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [References](#references)
- [License](#license)
- [Author](#author)

---

## Project Overview

This project leverages machine learning to predict the likelihood of heart disease based on medical data. It uses the Cleveland dataset and implements three machine learning algorithms: Support Vector Machine (SVM), TabNet, and Random Forest. The application is built using Django for the web interface and includes data visualization for better insights.

---

## Features

- **Machine Learning Models**: Utilizes SVM, TabNet, and Random Forest for accurate predictions.
- **Web Interface**: A user-friendly Django-based interface for inputting medical data.
- **Data Visualization**: Visual representations of data for enhanced understanding.
- **Static and Media Management**: Efficient handling of static and media files via Django settings.

---

## Project Structure

The project is organized as follows:

- `.gitignore`
- `2022-09-12 SLIDE Heart disease prediction using ML.pptx`
- `Certificate _ Index.pdf`
- `Documentation.pdf`
- `HeartAttack.csv`
- `Project 17_01_2023.html`
- `Project 28_12_2022.ipynb`
- `README.md`
- `In Python Full Stack/`
  - `db.sqlite3`
  - `manage.py`
  - `requirements.txt`
  - `admin/`
    - `css/`
    - `img/`
    - `js/`
  - `app/`
    - `__init__.py`
    - `admin.py`
    - `apps.py`
    - `models.py`
    - `tests.py`
    - `urls.py`
    - `views.py`
    - `__pycache__/`
    - `migrations/`
  - `assets/`
    - `css/`
  - `HeartDiseasePredictionUsingML/`
    - `__init__.py`
    - `asgi.py`
    - `settings.py`
    - `urls.py`
    - `wsgi.py`
  - `static/`
    - `css/`
  - `templates/`
    - `result.html`

---

## Installation

Follow these steps to set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Heart_disease_prediction_using_ML.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd Heart_disease_prediction_using_ML/In\ Python\ Full\ Stack/
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Apply migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

6. **Open the application**:
   Navigate to `http://127.0.0.1:8000` in your browser.

---

## Dataset

The project uses the **Cleveland dataset** for heart disease prediction. Download it from:
[Kaggle: Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

---

## Files and Directories

### Key Files
- **HeartAttack.csv**: Dataset used for training and testing machine learning models.
- **Project 28_12_2022.ipynb**: Jupyter Notebook with model training and evaluation code.
- **Project 17_01_2023.html**: HTML file for presenting project results.
- **manage.py**: Django's command-line utility for administrative tasks.
- **settings.py**: Django configuration file for the application.

### Key Directories
- **app/**: Core Django application files, including models, views, and URLs.
- **templates/**: HTML templates for rendering web pages.
- **static/**: Static files such as CSS and JavaScript.
- **assets/**: Additional assets for the application.

---

## Usage

1. Access the web interface at `http://127.0.0.1:8000`.
2. Input the required medical data into the form.
3. Submit the form to receive a prediction about the likelihood of heart disease.
4. View the results on the prediction page.

---

## Technologies Used

- **Backend**: Python, Django
- **Frontend**: HTML, CSS
- **Machine Learning**: SVM, TabNet, Random Forest
- **Database**: SQLite

---

## References

- **Learn SVM**: [Javatpoint: Support Vector Machine](https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm)
- **Learn Random Forest**: [Javatpoint: Random Forest](https://www.javatpoint.com/machine-learning-random-forest-algorithm)
- **Learn TabNet**: [TabNet Paper](https://arxiv.org/pdf/1908.07442v5.pdf)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author

Hello 👋🏻 I'm an Aspiring IT Professional | Web Developer | Python Enthusiast | Game Developer. Successfully completed this project on heart disease prediction using machine learning.
