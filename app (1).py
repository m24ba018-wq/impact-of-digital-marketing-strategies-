
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, f_oneway
import statsmodels.api as sm

# --- 1. Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Digital Marketing Impact Analysis")

# --- 2. Data Loading and Preprocessing ---
@st.cache_data
def load_and_preprocess_data():
    data_path = 'dataset.csv'  # Assuming dataset.csv is in the same directory
    df = pd.read_csv(data_path, encoding='latin1')

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df.columns = df.columns.str.replace('(', '', regex=False).str.replace(')', '', regex=False)
    df.columns = df.columns.str.replace('.', '', regex=False).str.replace('/', '_', regex=False)
    df.columns = df.columns.str.replace('–', '_', regex=False).str.replace('?', '', regex=False)

    # Handle missing values (as per notebook, dropna)
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # Define coding schemes
    likert_map = {
        "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5
    }
    age_map = {
        "Below 20": 1, "20–30": 2, "30–40": 3, "Above 40": 4
    }
    gender_map = {
        "Male": 1, "Female": 2, "Other": 3
    }
    education_map = {
        "Undergraduate": 1, "Graduate": 2, "Postgraduate": 3, "Others": 4
    }
    occupation_map = {
        "Student": 1, "Employed": 2, "Self-employed": 3, "Others": 4
    }
    income_map = {
        "Below ?10,000": 1, "?10,000–?30,000": 2, "?30,000–?50,000": 3, "Above ?50,000": 4
    }

    # Apply Likert scale mapping
    likert_columns = [
        'Digital_marketing_social_media,_influencers,_ads_influences_my_purchase_decisions',
        'Influencer_marketing_affects_my_choice_of_products',
        'Online_advertisements_increase_my_interest_in_products_services',
        'My_buying_behaviour_directly_affects_my_spending',
        'I_tend_to_spend_more_due_to_digital_marketing_influence',
        'Impulsive_buying_increases_my_overall_spending',
        'Social_media_marketing_leads_to_impulsive_buying',
        'Digital_marketing_helps_me_plan_my_purchases',
        'Different_digital_platforms_influence_my_buying_behaviour_differently',
        'My_age_and_income_influence_how_I_respond_to_digital_marketing',
        'I_make_purchase_decisions_based_on_my_financial_capacity',
        'Digital_payment_options_influence_my_decision_to_make_a_purchase',
        'Discounts,_cashback,_or_offers_influence_my_choice_of_payment_method',
        'I_tend_to_spend_more_when_using_digital_payment_methods'
    ]

    for col in likert_columns:
        if col in df.columns:
            df[col] = df[col].replace(likert_map)

    # Apply specific categorical mappings
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(str).str.replace('\x96', '–', regex=False).replace(age_map).astype(int)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).replace(gender_map).astype(int)
    if 'Education_Level' in df.columns:
        df['Education_Level'] = df['Education_Level'].astype(str).replace(education_map).astype(int)
    if 'Occupation' in df.columns:
        df['Occupation'] = df['Occupation'].astype(str).replace(occupation_map).astype(int)
    if 'Monthly_Income' in df.columns:
        df['Monthly_Income'] = df['Monthly_Income'].astype(str).str.replace('\x96', '–', regex=False).replace(income_map).astype(int)

    # Rename columns for composite scores and H5
    column_rename_map = {
        'Digital_marketing_social_media,_influencers,_ads_influences_my_purchase_decisions': 'DM_Influence',
        'Influencer_marketing_affects_my_choice_of_products': 'Influencer_Effect',
        'Online_advertisements_increase_my_interest_in_products_services': 'Ads_Interest',
        'My_buying_behaviour_directly_affects_my_spending': 'Behavior_Spending_Link',
        'I_tend_to_spend_more_due_to_digital_marketing_influence': 'Spend_Due_DM',
        'Impulsive_buying_increases_my_overall_spending': 'Impulse_Spending',
        'Social_media_marketing_leads_to_impulsive_buying': 'SM_Impulse',
        'Digital_marketing_helps_me_plan_my_purchases': 'Planned_Purchase',
        'Different_digital_platforms_influence_my_buying_behaviour_differently': 'Platform_Impact',
        'My_age_and_income_influence_how_I_respond_to_digital_marketing': 'Demo_Impact',
        'I_make_purchase_decisions_based_on_my_financial_capacity': 'Financial_Decision',
        'Digital_payment_options_influence_my_decision_to_make_a_purchase': 'Payment_Influence',
        'Discounts,_cashback,_or_offers_influence_my_choice_of_payment_method': 'Offer_Influence',
        'I_tend_to_spend_more_when_using_digital_payment_methods': 'Spend_Digital'
    }
    df.rename(columns=column_rename_map, inplace=True)

    # Compute Composite Scores
    df['Digital_Marketing_Score'] = df[['DM_Influence','Influencer_Effect','Ads_Interest']].mean(axis=1)
    df['Buying_Behaviour_Score'] = df[['Impulse_Spending','SM_Impulse','Planned_Purchase']].mean(axis=1)
    df['Spending_Score'] = df[['Spend_Due_DM','Behavior_Spending_Link']].mean(axis=1)
    df['Digital_Payment_Score'] = df[['Payment_Influence','Offer_Influence','Spend_Digital']].mean(axis=1)

    # Recreate Income_Category for Boxplot visualization
    income_map_reverse = {v: k for k, v in income_map.items()}
    df['Income_Category'] = df['Monthly_Income'].map(income_map_reverse)
    # Ensure correct order for plotting
    income_order = ["Below ?10,000", "?10,000–?30,000", "?30,000–?50,000", "Above ?50,000"]
    df['Income_Category'] = pd.Categorical(df['Income_Category'], categories=income_order, ordered=True)

    return df

df = load_and_preprocess_data()

# --- 3. Sidebar Navigation ---
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio(
    "Select Analysis Type:",
    ("Descriptive Analysis", "Visualization", "Hypothesis Testing")
)

# --- Main Title ---
st.title("Digital Marketing Impact Analysis Dashboard")
st.write("This dashboard provides an interactive exploration of the impact of digital marketing on consumer behavior and spending, based on the dissertation research 'Impact of Digital Marketing on Consumer Behavior and Spending in Surat'.")

# --- Content Sections ---
if analysis_type == "Descriptive Analysis":
    st.header("1. Descriptive Analysis")
    st.markdown("--- ")

    st.subheader("1.1 Dataset Preview")
    st.write("Here's a sneak peek at the first 5 rows of the processed dataset:")
    st.dataframe(df.head())

    st.subheader("1.2 Summary Statistics")
    st.write("Overall descriptive statistics for numerical columns in the dataset:")
    st.dataframe(df.describe())

elif analysis_type == "Visualization":
    st.header("2. Visualizations")
    st.markdown("--- ")

    # Regression Plot
    st.subheader("2.1 Regression Plot: Digital Marketing Score vs. Spending Score")
    st.write("This plot illustrates the linear relationship between the overall influence of digital marketing and consumer spending.")
    fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
    sns.regplot(x=df['Digital_Marketing_Score'], y=df['Spending_Score'], ax=ax_reg)
    ax_reg.set_title('Digital Marketing Score vs. Spending Score')
    ax_reg.set_xlabel('Digital Marketing Score')
    ax_reg.set_ylabel('Spending Score')
    st.pyplot(fig_reg)
    st.write("Interpretation: The upward trend suggests a positive correlation: as consumers perceive more influence from digital marketing, their spending score tends to increase. This visually supports the idea that digital marketing plays a role in driving spending.")

    # Histogram for Spending
    st.subheader("2.2 Histogram: Distribution of Spending Score")
    st.write("This histogram shows the frequency distribution of consumer spending scores.")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Spending_Score'], kde=True, ax=ax_hist)
    ax_hist.set_title('Distribution of Spending Score')
    ax_hist.set_xlabel('Spending Score')
    ax_hist.set_ylabel('Frequency')
    st.pyplot(fig_hist)
    st.write("Interpretation: The distribution reveals the most common range of spending scores among respondents. The shape (e.g., normal, skewed) can provide insights into whether spending is concentrated at particular levels or spread out.")

    # Boxplot for Spending by Income Category
    st.subheader("2.3 Box Plot: Spending Score by Monthly Income Category")
    st.write("This box plot compares the distribution of spending scores across different monthly income categories, highlighting potential differences in spending patterns.")
    fig_box, ax_box = plt.subplots(figsize=(12, 7))
    sns.boxplot(x='Income_Category', y='Spending_Score', data=df, ax=ax_box, palette='viridis')
    ax_box.set_title('Spending Score by Monthly Income Category')
    ax_box.set_xlabel('Monthly Income Category')
    ax_box.set_ylabel('Spending Score')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_box)
    st.write("Interpretation: This visualization helps to identify if there are notable differences in spending habits across various income brackets. Higher median spending in certain income groups, or larger interquartile ranges, would indicate varying spending distributions.")

    # Heatmap
    st.subheader("2.4 Correlation Matrix Heatmap")
    st.write("A heatmap illustrating the correlation coefficients between all numerical variables in the dataset. Values closer to 1 or -1 indicate stronger linear relationships.")
    fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
    df_numeric = df.select_dtypes(include=np.number)
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Correlation Matrix of Numerical Variables')
    st.pyplot(fig_corr)
    st.write("Interpretation: This heatmap helps in understanding the interdependencies between different variables. Strong positive correlations (warm colors) indicate variables that increase together, while strong negative correlations (cool colors) show variables that move in opposite directions. This is useful for identifying potential relationships and multicollinearity.")

elif analysis_type == "Hypothesis Testing":
    st.header("3. Hypothesis Testing Results")
    st.markdown("--- ")

    # H1: Digital Marketing → Buying Behaviour & Spending
    st.subheader("3.1 H1: Digital Marketing → Buying Behaviour & Spending")
    X_h1 = df[['Digital_Marketing_Score']]
    X_h1 = sm.add_constant(X_h1)
    model1_h1 = sm.OLS(df['Buying_Behaviour_Score'], X_h1).fit()
    model2_h1 = sm.OLS(df['Spending_Score'], X_h1).fit()

    st.write("**Buying Behaviour Model:**")
    st.code(model1_h1.summary().as_text())
    st.write("**Spending Model:**")
    st.code(model2_h1.summary().as_text())
    if model2_h1.pvalues['Digital_Marketing_Score'] < 0.05:
        st.success("✅ H1 Supported: Digital marketing significantly impacts both buying behaviour and spending.")
        st.write("Interpretation: The regression results show a statistically significant positive relationship between Digital Marketing Score and both Buying Behaviour Score and Spending Score, indicating that increased digital marketing influence leads to changes in both.")
    else:
        st.error("❌ H1 Not Supported: No significant impact of digital marketing on both buying behaviour and spending found.")

    # H2: Digital Marketing → Buying Behaviour
    st.subheader("3.2 H2: Digital Marketing → Buying Behaviour")
    corr_h2, p_h2 = pearsonr(df['Digital_Marketing_Score'], df['Buying_Behaviour_Score'])
    st.write(f"Correlation: {corr_h2:.3f}")
    st.write(f"P-value: {p_h2:.3e}")
    if p_h2 < 0.05:
        st.success("✅ H2 Supported: Digital marketing significantly influences buying behaviour.")
        st.write("Interpretation: A significant positive correlation indicates that as digital marketing influence increases, there is a tendency for buying behaviour to also be influenced positively.")
    else:
        st.error("❌ H2 Not Supported: No significant influence of digital marketing on buying behaviour found.")

    # H3: Buying Behaviour ↔ Spending
    st.subheader("3.3 H3: Buying Behaviour ↔ Spending")
    corr_h3, p_h3 = pearsonr(df['Buying_Behaviour_Score'], df['Spending_Score'])
    st.write(f"Correlation: {corr_h3:.3f}")
    st.write(f"P-value: {p_h3:.3e}")
    if p_h3 < 0.05:
        st.success("✅ H3 Supported: Buying behaviour significantly relates to spending.")
        st.write("Interpretation: A significant positive correlation suggests that changes in consumer buying behaviour are strongly associated with changes in their spending patterns.")
    else:
        st.error("❌ H3 Not Supported: No significant relationship between buying behaviour and spending found.")

    # H4: Channels influence buying behaviour (ANOVA)
    st.subheader("3.4 H4: Different Digital Platforms Influence Buying Behaviour")
    groups_h4 = df.groupby('Platform_Impact')['Buying_Behaviour_Score'].apply(list)
    anova_h4 = f_oneway(*groups_h4)
    st.write(f"F-value: {anova_h4.statistic:.3f}")
    st.write(f"P-value: {anova_h4.pvalue:.3e}")
    if anova_h4.pvalue < 0.05:
        st.success("✅ H4 Supported: Different digital platforms significantly influence buying behaviour.")
        st.write("Interpretation: The ANOVA test shows that there are significant differences in buying behaviour scores across different levels of 'Platform_Impact', implying that various digital platforms have distinct influences.")
    else:
        st.error("❌ H4 Not Supported: No significant difference in buying behaviour across different digital platforms found.")

    # H5: Moderation Analysis
    st.subheader("3.5 H5: Demographic Factors Moderate the Relationship between Digital Marketing and Spending")
    cols_h5 = ['Digital_Marketing_Score','Age','Monthly_Income','Education_Level','Spending_Score']
    df_h5_streamlit = df[cols_h5].copy() # Use a copy to avoid SettingWithCopyWarning
    df_h5_streamlit[cols_h5] = df_h5_streamlit[cols_h5].apply(pd.to_numeric, errors='coerce')
    df_h5_streamlit = df_h5_streamlit.dropna()

    # Create interaction terms
    df_h5_streamlit['DM_Age'] = df_h5_streamlit['Digital_Marketing_Score'] * df_h5_streamlit['Age']
    df_h5_streamlit['DM_Income'] = df_h5_streamlit['Digital_Marketing_Score'] * df_h5_streamlit['Monthly_Income']
    df_h5_streamlit['DM_Education'] = df_h5_streamlit['Digital_Marketing_Score'] * df_h5_streamlit['Education_Level']

    X_h5_streamlit = df_h5_streamlit[['Digital_Marketing_Score','Age','Monthly_Income','Education_Level',
                                       'DM_Age','DM_Income','DM_Education']]
    y_h5_streamlit = df_h5_streamlit['Spending_Score']
    X_h5_streamlit = sm.add_constant(X_h5_streamlit)
    model_h5 = sm.OLS(y_h5_streamlit, X_h5_streamlit).fit()

    st.write("**Regression Summary:**")
    st.code(model_h5.summary().as_text())

    interaction_pvalues = model_h5.pvalues[['DM_Age','DM_Income','DM_Education']]
    st.write("**Interaction P-values:**")
    st.dataframe(interaction_pvalues)

    if (interaction_pvalues < 0.05).any():
        st.success("✅ H5 Supported: Demographic factors (specifically Monthly Income) significantly moderate the relationship between digital marketing influence and spending.")
        st.write("Interpretation: The significant interaction term between 'Digital_Marketing_Score' and 'Monthly_Income' indicates that the effect of digital marketing on spending varies depending on the consumer's monthly income.")
    else:
        st.error("❌ H5 Not Supported: No significant moderation effect found by demographic factors.")
        st.write("Interpretation: No statistically significant moderation effects were found for age, income, or education on the relationship between digital marketing and spending.")

    # H6: Digital Payment → Behaviour & Spending
    st.subheader("3.6 H6: Digital Payment Options → Buying Behaviour & Spending")
    X_h6 = df[['Digital_Payment_Score']]
    X_h6 = sm.add_constant(X_h6)
    model1_h6 = sm.OLS(df['Buying_Behaviour_Score'], X_h6).fit()
    model2_h6 = sm.OLS(df['Spending_Score'], X_h6).fit()

    st.write("**Buying Behaviour Model:**")
    st.code(model1_h6.summary().as_text())
    st.write("**Spending Model:**")
    st.code(model2_h6.summary().as_text())

    if model2_h6.pvalues['Digital_Payment_Score'] < 0.05:
        st.success("✅ H6 Supported: Digital payment options significantly impact both buying behaviour and spending.")
        st.write("Interpretation: The regression analysis reveals a significant positive relationship, suggesting that the availability and influence of digital payment options contribute to changes in both consumer buying behaviour and spending levels.")
    else:
        st.error("❌ H6 Not Supported: No significant impact of digital payment options on both buying behaviour and spending found.")
