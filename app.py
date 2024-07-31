# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import base64


# app = Flask(__name__,template_folder='templates')


# # Load ESG dataset
# esg_data = pd.read_excel('datasets/esg.xlsx')

# # Check if 'name' column exists and is correctly loaded
# if 'name' not in esg_data.columns:
#     raise KeyError("The dataset does not contain the 'name' column, which is required.")

# # Mock carbon footprint and additional financial data
# esg_data['carbon_footprint'] = np.random.uniform(100, 1000, len(esg_data))
# esg_data['ROI'] = np.random.uniform(0, 0.15, len(esg_data))
# esg_data['Beta'] = np.random.uniform(-1, 3, len(esg_data))
# esg_data['P/E_Ratio'] = np.random.uniform(10, 30, len(esg_data))

# # Normalize scores
# scaler = MinMaxScaler()
# esg_data[['environment_score', 'social_score', 'governance_score', 'total_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio']] = scaler.fit_transform(
#     esg_data[['environment_score', 'social_score', 'governance_score', 'total_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio']]
# )

# weights = {
#     'environment_score': 0.30,
#     'social_score': 0.20,
#     'governance_score': 0.20,
#     'carbon_footprint': 0.10,
#     'ROI': 0.20,
#     'Beta': 0.05,
#     'P/E_Ratio': 0.05
# }

# # Calculate total_score with inverse adjustments for metrics where lower is better
# esg_data['total_score'] = (
#     weights['environment_score'] * esg_data['environment_score'] +
#     weights['social_score'] * esg_data['social_score'] +
#     weights['governance_score'] * esg_data['governance_score'] +
#     weights['carbon_footprint'] * (1 - esg_data['carbon_footprint']) +  # Assuming lower carbon footprint is better
#     weights['ROI'] * esg_data['ROI'] +
#     weights['Beta'] * (1 - esg_data['Beta']) +  # Assuming lower Beta is better (less risk)
#     weights['P/E_Ratio'] * (1 / esg_data['P/E_Ratio']+1e-6)  # Assuming lower P/E Ratio is better
# )


# @app.route("/", methods=["GET", "POST"])
# def index():
#     recommendations = None
#     chart_url = None
#     if request.method == "POST":
#         # Retrieve user inputs from the form
#         investment_amount = float(request.form.get('investment_amount'))
#         risk_tolerance = request.form.get('risk_tolerance')
#         environment_importance = request.form.get('environment_importance')
#         social_importance = request.form.get('social_importance')
#         governance_importance = request.form.get('governance_importance')
#         carbon_footprint_importance = request.form.get('carbon_footprint_importance')
#         roi_importance = request.form.get('roi_importance')
#         beta_importance = request.form.get('beta_importance')
#         pe_ratio_importance = request.form.get('pe_ratio_importance')

#         # Define filters for the financial metrics based on user input
#         esg_data_filtered = esg_data.copy()  # Start with a full copy of the data
        
#         if roi_importance == 'High':
#             esg_data_filtered = esg_data_filtered[esg_data_filtered['ROI'] > 0.07]
#         elif roi_importance == 'Low':
#             esg_data_filtered = esg_data_filtered[esg_data_filtered['ROI'] <= 0.07]

#         if beta_importance == 'Ideal':
#             esg_data_filtered = esg_data_filtered[(esg_data_filtered['Beta'] >= 0.8) & (esg_data_filtered['Beta'] <= 1.2)]
#         elif beta_importance == 'Low Risk':
#             esg_data_filtered = esg_data_filtered[esg_data_filtered['Beta'] < 0.8]
#         elif beta_importance == 'High Risk':
#             esg_data_filtered = esg_data_filtered[esg_data_filtered['Beta'] > 1.2]

#         if pe_ratio_importance == 'Low':
#             esg_data_filtered = esg_data_filtered[esg_data_filtered['P/E_Ratio'] < 23]
#         elif pe_ratio_importance == 'High':
#             esg_data_filtered = esg_data_filtered[esg_data_filtered['P/E_Ratio'] >= 23]

#         # Convert importance to numerical values
#         def get_importance_value(importance, high=0.5, medium=0.2):
#             return high if importance == 'Very Important' else medium if importance == 'Somewhat Important' else 0

#         user_preferences = {
#             'environment_importance': get_importance_value(environment_importance),
#             'social_importance': get_importance_value(social_importance),
#             'governance_importance': get_importance_value(governance_importance),
#             'carbon_footprint_importance': get_importance_value(carbon_footprint_importance, 0.2, 0.1)
#         }

#         user_profile = create_user_profile(user_preferences)
#         recommendations = recommend_investments(user_profile, esg_data_filtered)
#         chart_url = generate_chart(recommendations, 'total_score', 'Total Score', 'Top Investment Recommendations')
#         carbon_chart_url = generate_chart(recommendations, 'carbon_footprint', 'Carbon Footprint Reduction', 'Carbon Footprint Impact')
#         roi_chart_url = generate_chart(recommendations, 'ROI', 'Expected ROI', 'Profitability Impact')

#     return render_template("index.html", recommendations=recommendations, chart_url=chart_url, carbon_chart_url=carbon_chart_url, roi_chart_url=roi_chart_url)



# def create_user_profile(preferences):
#     # Create a DataFrame that matches the structure of esg_data for the similarity calculation
#     return pd.DataFrame({
#         'environment_score': [preferences['environment_importance']],
#         'social_score': [preferences['social_importance']],
#         'governance_score': [preferences['governance_importance']],
#         'carbon_footprint': [preferences['carbon_footprint_importance']],
#         'ROI': [1.0],  # Assuming ROI is maximally important
#         'Beta': [1.0],  # Assuming Beta is maximally important
#         'P/E_Ratio': [1.0],  # Assuming P/E Ratio is maximally important
#         'total_score': [1.0]  # Assuming the total score importance is maximum
#     })

# def recommend_investments(user_profile, esg_data):
#     # Ensure both dataframes have the same columns in the same order
#     user_profile = user_profile.reindex(columns=['environment_score', 'social_score', 'governance_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio', 'total_score'])
#     esg_data = esg_data.reindex(columns=['name', 'environment_score', 'social_score', 'governance_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio', 'total_score'])
    

#     recommendations = esg_data.sort_values('total_score', ascending=False).head(5)
    
#     return recommendations[['name', 'environment_score', 'social_score', 'governance_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio', 'total_score']]

# def generate_chart(recommendations, metric, ylabel, title):
#     # Plotting the recommendations' metrics
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='name', y=metric, data=recommendations, palette='viridis')
#     plt.title(title)
#     plt.ylabel(ylabel)
#     plt.xlabel('Company Name')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()

#     # Convert plot to PNG image and encode in base64
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plot_url = base64.b64encode(img.getvalue()).decode('UTF-8')
#     plt.close()
#     return plot_url



# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__, template_folder='templates')

# Load ESG dataset
esg_data = pd.read_excel('datasets/esg.xlsx')

# Check if 'name' column exists and is correctly loaded
if 'name' not in esg_data.columns:
    raise KeyError("The dataset does not contain the 'name' column, which is required.")

# Mock carbon footprint and additional financial data
esg_data['carbon_footprint'] = np.random.uniform(100, 1000, len(esg_data))
esg_data['ROI'] = np.random.uniform(0, 0.15, len(esg_data))
esg_data['Beta'] = np.random.uniform(-1, 3, len(esg_data))
esg_data['P/E_Ratio'] = np.random.uniform(10, 30, len(esg_data))

# Normalize scores
scaler = MinMaxScaler()
esg_data[['environment_score', 'social_score', 'governance_score', 'total_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio']] = scaler.fit_transform(
    esg_data[['environment_score', 'social_score', 'governance_score', 'total_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio']]
)

weights = {
    'environment_score': 0.30,
    'social_score': 0.20,
    'governance_score': 0.20,
    'carbon_footprint': 0.10,
    'ROI': 0.20,
    'Beta': 0.05,
    'P/E_Ratio': 0.05
}

# Calculate total_score with inverse adjustments for metrics where lower is better
esg_data['total_score'] = (
    weights['environment_score'] * esg_data['environment_score'] +
    weights['social_score'] * esg_data['social_score'] +
    weights['governance_score'] * esg_data['governance_score'] +
    weights['carbon_footprint'] * (1 - esg_data['carbon_footprint']) +  # Assuming lower carbon footprint is better
    weights['ROI'] * esg_data['ROI'] +
    weights['Beta'] * (1 - esg_data['Beta']) +  # Assuming lower Beta is better (less risk)
    weights['P/E_Ratio'] * (1 / (esg_data['P/E_Ratio'] + 1e-6))  # Assuming lower P/E Ratio is better
)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    chart_url = None
    carbon_chart_url = None
    roi_chart_url = None
    beta_chart_url = None

    if request.method == "POST":
        # Retrieve user inputs from the form
        investment_amount = float(request.form.get('investment_amount'))
        risk_tolerance = request.form.get('risk_tolerance')
        environment_importance = request.form.get('environment_importance')
        social_importance = request.form.get('social_importance')
        governance_importance = request.form.get('governance_importance')
        carbon_footprint_importance = request.form.get('carbon_footprint_importance')
        roi_importance = request.form.get('roi_importance')
        beta_importance = request.form.get('beta_importance')
        pe_ratio_importance = request.form.get('pe_ratio_importance')

        # Define filters for the financial metrics based on user input
        esg_data_filtered = esg_data.copy()  # Start with a full copy of the data
        
        if roi_importance == 'High':
            esg_data_filtered = esg_data_filtered[esg_data_filtered['ROI'] > 0.07]
        elif roi_importance == 'Low':
            esg_data_filtered = esg_data_filtered[esg_data_filtered['ROI'] <= 0.07]

        if beta_importance == 'Ideal':
            esg_data_filtered = esg_data_filtered[(esg_data_filtered['Beta'] >= 0.8) & (esg_data_filtered['Beta'] <= 1.2)]
        elif beta_importance == 'Low Risk':
            esg_data_filtered = esg_data_filtered[esg_data_filtered['Beta'] < 0.8]
        elif beta_importance == 'High Risk':
            esg_data_filtered = esg_data_filtered[esg_data_filtered['Beta'] > 1.2]

        if pe_ratio_importance == 'Low':
            esg_data_filtered = esg_data_filtered[esg_data_filtered['P/E_Ratio'] < 23]
        elif pe_ratio_importance == 'High':
            esg_data_filtered = esg_data_filtered[esg_data_filtered['P/E_Ratio'] >= 23]

        # Convert importance to numerical values
        def get_importance_value(importance, high=0.5, medium=0.2):
            return high if importance == 'Very Important' else medium if importance == 'Somewhat Important' else 0

        user_preferences = {
            'environment_importance': get_importance_value(environment_importance),
            'social_importance': get_importance_value(social_importance),
            'governance_importance': get_importance_value(governance_importance),
            'carbon_footprint_importance': get_importance_value(carbon_footprint_importance, 0.2, 0.1)
        }

        user_profile = create_user_profile(user_preferences)
        recommendations = recommend_investments(user_profile, esg_data_filtered)
        chart_url = generate_chart(recommendations, 'total_score', 'Total Score', 'Top Investment Recommendations')
        carbon_chart_url = generate_chart(recommendations, 'carbon_footprint', 'Carbon Footprint Reduction', 'Carbon Footprint Impact')
        roi_chart_url = generate_chart(recommendations, 'ROI', 'Expected ROI', 'Profitability Impact')
        beta_chart_url = generate_chart(recommendations, 'Beta', 'Beta Values', 'Risk Analysis')

    return render_template("index.html", recommendations=recommendations, chart_url=chart_url, carbon_chart_url=carbon_chart_url, roi_chart_url=roi_chart_url, beta_chart_url=beta_chart_url)

@app.route("/blogs")
def blogs():
    return render_template("blogs.html")

def create_user_profile(preferences):
    return pd.DataFrame({
        'environment_score': [preferences['environment_importance']],
        'social_score': [preferences['social_importance']],
        'governance_score': [preferences['governance_importance']],
        'carbon_footprint': [preferences['carbon_footprint_importance']],
        'ROI': [1.0],  # Assuming ROI is maximally important
        'Beta': [1.0],  # Assuming Beta is maximally important
        'P/E_Ratio': [1.0],  # Assuming P/E Ratio is maximally important
        'total_score': [1.0]  # Assuming the total score importance is maximum
    })

def recommend_investments(user_profile, esg_data):
    user_profile = user_profile.reindex(columns=['environment_score', 'social_score', 'governance_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio', 'total_score'])
    esg_data = esg_data.reindex(columns=['name', 'environment_score', 'social_score', 'governance_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio', 'total_score'])
    
    recommendations = esg_data.sort_values('total_score', ascending=False).head(5)
    
    return recommendations[['name', 'environment_score', 'social_score', 'governance_score', 'carbon_footprint', 'ROI', 'Beta', 'P/E_Ratio', 'total_score']]

def generate_chart(recommendations, metric, ylabel, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='name', y=metric, data=recommendations, palette='viridis')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Company Name')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('UTF-8')
    plt.close()
    return plot_url

if __name__ == "__main__":
    app.run(debug=True)
