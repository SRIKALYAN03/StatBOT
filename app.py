from flask import Flask, render_template, request, jsonify
import json
import pyttsx3
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.stats.weightstats import ztest
import io
import base64
import os
import threading
import subprocess

app = Flask(__name__)

# Load Statistical Definitions JSON
with open("statistics_dataset.json", "r") as file:
    statistics_dataset = json.load(file)

# Initialize Jarvis Voice
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Male Jarvis-style voice



speech_lock = threading.Lock()

def speak(text):
    """Call external Python script to safely handle speech"""
    try:
        subprocess.Popen(['python', 'speaker.py', text])
        print("Jarvis:", text)
    except Exception as e:
        print(f"Voice Error: {e}")




def get_statistical_answer(question):
    """Retrieve definition, formula, and example of a statistical term."""
    for entry in statistics_dataset:
        if entry["topic"].lower() in question.lower():
            answer = f"{entry['definition']} The formula is: {entry['formula']}. Example: {entry['example']}."
            return answer
    return "Sorry, I couldn't find an answer to that question. Please ask again."

def load_file(file):
    """Load CSV or Excel file and return DataFrame or error message"""
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return None, "Invalid file type. Please upload a CSV or Excel file."
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def create_plot():
    """Helper function to create and encode plot"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Statistical Analysis Functions
def central_tendency(df, column):
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode().iloc[0] if not df[column].mode().empty else "No mode"
    return f"Mean: {mean:.4f}, Median: {median:.4f}, Mode: {mode}"

def dispersion(df, column):
    variance = df[column].var()
    std_dev = df[column].std()
    return f"Variance: {variance:.4f}, Standard Deviation: {std_dev:.4f}"

def quartiles_deciles_percentiles(df, column):
    q1 = df[column].quantile(0.25)
    q2 = df[column].quantile(0.50)
    q3 = df[column].quantile(0.75)
    percentiles = [round(df[column].quantile(p), 4) for p in np.arange(0.1, 1.0, 0.1)]
    return (f"Quartiles -> Q1: {q1:.4f}, Median: {q2:.4f}, Q3: {q3:.4f}\n"
            f"Deciles & Percentiles: {percentiles}")

def skewness_kurtosis(df, column):
    skewness = df[column].skew()
    kurtosis = df[column].kurt()
    return f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}"

def normal_distribution(df, column):
    stat, p = stats.shapiro(df[column])
    is_normal = "Data is normally distributed." if p > 0.05 else "Data is NOT normally distributed."
    return f"Shapiro-Wilk Test Statistic: {stat:.4f}, P-Value: {p:.4f}\n{is_normal}"

def small_sample_test(df, column):
    t_stat, p_value = stats.ttest_1samp(df[column], df[column].mean())
    return f"T-Test Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}"

def large_sample_test(df, column):
    z_stat, p_value = ztest(df[column])
    return f"Z-Test Statistic: {z_stat:.4f}, P-Value: {p_value:.4f}"

def chi_square_test(df, column):
    observed = df[column].value_counts()
    chi2, p = stats.chisquare(observed)
    return f"Chi-Square Statistic: {chi2:.4f}, P-Value: {p:.4f}"

def two_sample_t_test(df, col1, col2):
    t_stat, p_value = stats.ttest_ind(df[col1], df[col2])
    return f"Two-Sample T-Test: T-Statistic = {t_stat:.4f}, P-Value = {p_value:.4f}"

def wilcoxon_test(df, col1, col2):
    stat, p_value = stats.wilcoxon(df[col1], df[col2])
    return f"Wilcoxon Test: Statistic = {stat:.4f}, P-Value = {p_value:.4f}"

def mann_whitney_test(df, col1, col2):
    stat, p_value = stats.mannwhitneyu(df[col1], df[col2])
    return f"Mann-Whitney U Test: Statistic = {stat:.4f}, P-Value = {p_value:.4f}"

def kruskal_wallis_test(df, cols):
    stat, p_value = stats.kruskal(*(df[col] for col in cols))
    return f"Kruskal-Wallis Test: Statistic = {stat:.4f}, P-Value = {p_value:.4f}"

def one_way_anova(df, cols):
    stat, p_value = stats.f_oneway(*(df[col] for col in cols))
    return f"One-Way ANOVA: F-Statistic = {stat:.4f}, P-Value = {p_value:.4f}"

def levenes_test(df, cols):
    stat, p_value = stats.levene(*(df[col] for col in cols))
    return f"Levene's Test: Statistic = {stat:.4f}, P-Value = {p_value:.4f}"

def covariance(df, col1, col2):
    covar = df[col1].cov(df[col2])
    return f"Covariance between {col1} and {col2} is {covar:.4f}"

def correlation(df, col1, col2):
    corr = df[col1].corr(df[col2])
    return f"Correlation coefficient between {col1} and {col2} is {corr:.4f}"

def regression(df, x_col, y_col):
    X = df[[x_col]]
    y = df[y_col]
    model = LinearRegression()
    model.fit(X, y)
    coef = model.coef_[0]
    intercept = model.intercept_
    return f"Regression equation: {y_col} = {intercept:.4f} + {coef:.4f} * {x_col}"

# Visualization Functions
def scatter_plot(df, x_col, y_col):
    plt.figure()
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    return create_plot()

def pie_chart(df, column):
    plt.figure()
    df[column].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f"Pie Chart of {column}")
    return create_plot()

def box_plot(df, column):
    plt.figure()
    sns.boxplot(y=df[column])
    plt.title(f"Box Plot of {column}")
    return create_plot()

def bar_chart(df, column):
    plt.figure()
    df[column].value_counts().plot(kind='bar')
    plt.title(f"Bar Chart of {column}")
    return create_plot()

def heatmap(df):
    """Generate heatmap only for numeric columns"""
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    if corr_matrix.isnull().values.all():
        raise ValueError("Correlation matrix is empty or contains only NaN values.")

    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap of Correlations")
    return create_plot()


def violin_plot(df, column):
    plt.figure()
    sns.violinplot(y=df[column])
    plt.title(f"Violin Plot of {column}")
    return create_plot()

def swarm_plot(df, column):
    plt.figure()
    sns.swarmplot(y=df[column])
    plt.title(f"Swarm Plot of {column}")
    return create_plot()

def joint_plot(df, x_col, y_col):
    plt.figure()
    sns.jointplot(x=df[x_col], y=df[y_col], kind="scatter")
    return create_plot()

def pair_plot(df):
    plt.figure()
    sns.pairplot(df)
    return create_plot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = get_statistical_answer(question)
    speak(answer)
    return jsonify({'answer': answer})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    df, error = load_file(file)
    if error:
        return jsonify({'error': error})
    
    analysis_type = request.form['analysis_type']
    result = {}
    
    try:
        if analysis_type in ['central tendency', 'dispersion', 'quartiles', 
                            'skewness', 'normal distribution', 'small sample test',
                            'large sample test', 'chi-square test', 'box plot',
                            'violin plot', 'swarm plot', 'bar chart', 'pie chart']:
            column = request.form['column']
            if analysis_type == 'central tendency':
                result['text'] = central_tendency(df, column)
            elif analysis_type == 'dispersion':
                result['text'] = dispersion(df, column)
            elif analysis_type == 'quartiles':
                result['text'] = quartiles_deciles_percentiles(df, column)
            elif analysis_type == 'skewness':
                result['text'] = skewness_kurtosis(df, column)
            elif analysis_type == 'normal distribution':
                result['text'] = normal_distribution(df, column)
            elif analysis_type == 'small sample test':
                result['text'] = small_sample_test(df, column)
            elif analysis_type == 'large sample test':
                result['text'] = large_sample_test(df, column)
            elif analysis_type == 'chi-square test':
                result['text'] = chi_square_test(df, column)
            elif analysis_type == 'box plot':
                result['plot'] = box_plot(df, column)
            elif analysis_type == 'violin plot':
                result['plot'] = violin_plot(df, column)
            elif analysis_type == 'swarm plot':
                result['plot'] = swarm_plot(df, column)
            elif analysis_type == 'bar chart':
                result['plot'] = bar_chart(df, column)
            elif analysis_type == 'pie chart':
                result['plot'] = pie_chart(df, column)
        
        elif analysis_type in ['scatter plot', 'regression', 'covariance', 
                              'correlation', 'joint plot', 'two-sample t-test',
                              'wilcoxon test', 'mann-whitney test']:
            x_col = request.form['x_col']
            y_col = request.form['y_col']
            if analysis_type == 'scatter plot':
                result['plot'] = scatter_plot(df, x_col, y_col)
            elif analysis_type == 'regression':
                result['text'] = regression(df, x_col, y_col)
            elif analysis_type == 'covariance':
                result['text'] = covariance(df, x_col, y_col)
            elif analysis_type == 'correlation':
                result['text'] = correlation(df, x_col, y_col)
            elif analysis_type == 'joint plot':
                result['plot'] = joint_plot(df, x_col, y_col)
            elif analysis_type == 'two-sample t-test':
                result['text'] = two_sample_t_test(df, x_col, y_col)
            elif analysis_type == 'wilcoxon test':
                result['text'] = wilcoxon_test(df, x_col, y_col)
            elif analysis_type == 'mann-whitney test':
                result['text'] = mann_whitney_test(df, x_col, y_col)
        
        elif analysis_type in ['kruskal-wallis test', 'one-way anova', 'levenes test']:
            cols = request.form.getlist('cols[]')
            if analysis_type == 'kruskal-wallis test':
                result['text'] = kruskal_wallis_test(df, cols)
            elif analysis_type == 'one-way anova':
                result['text'] = one_way_anova(df, cols)
            elif analysis_type == 'levenes test':
                result['text'] = levenes_test(df, cols)
        
        elif analysis_type in ['heatmap', 'pair plot']:
            if analysis_type == 'heatmap':
                result['plot'] = heatmap(df)
            elif analysis_type == 'pair plot':
                result['plot'] = pair_plot(df)
        
        else:
            result['error'] = "Invalid analysis type selected"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f"Error during analysis: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)