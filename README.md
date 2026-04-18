# Digital Marketing Impact Analysis Dashboard

## Objective
This project provides an interactive dashboard to explore the impact of digital marketing strategies on consumer buying behavior and financial spending patterns, based on a dissertation research study conducted in Surat. The dashboard offers descriptive analysis, key visualizations, and a summary of hypothesis testing results.

## Dataset Description
The analysis is based on a survey dataset that has been cleaned and preprocessed. It includes the following key variables:
-   `Digital_Marketing_Score`: A composite score representing the overall influence of digital marketing (social media, influencers, ads) on purchase decisions.
-   `Buying_Behaviour_Score`: A composite score reflecting aspects of consumer buying behavior, including impulsivity and planning.
-   `Spending_Score`: A composite score indicating consumer spending habits influenced by digital marketing.
-   `Digital_Payment_Score`: A composite score related to the influence of digital payment options on purchases.
-   Demographic variables: `Age`, `Monthly_Income`, `Education_Level`, and `Occupation`.
-   `Platform_Impact`: A variable indicating how different digital platforms influence buying behavior.

## Features
-   **Descriptive Analysis**: View dataset preview and summary statistics.
-   **Visualization**: Explore key relationships through interactive plots:
    -   Regression plot: Digital Marketing Score vs. Spending Score
    -   Histogram: Distribution of Spending Score
    -   Box plot: Spending Score by Monthly Income Category
    -   Heatmap: Correlation Matrix of numerical variables
-   **Hypothesis Testing Results**: Review the outcomes of six hypotheses (H1-H6) related to digital marketing, buying behavior, and spending, complete with statistical summaries and interpretations.

## How to Run the Application
To run this Streamlit application locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Prepare your dataset:**
    Ensure your preprocessed dataset is named `dataset.csv` and placed in the root directory of the project. If your original dataset was named differently in Colab, please rename it to `dataset.csv`.

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

6.  **Access the Dashboard:**
    Your web browser will automatically open to the Streamlit app (usually at `http://localhost:8501`).

## Project Structure
```
Digital_Marketing_Impact_Analysis/
├── app.py
├── dataset.csv
├── requirements.txt
└── README.md
```

This structure ensures that your project is well-organized and ready for deployment or sharing on platforms like GitHub.
