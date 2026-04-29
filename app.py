import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import wbgapi as wb 
from statsmodels.tsa.arima.model import ARIMA
import shap
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# =============================================================================
# 1. SETUP & THEME
# =============================================================================

st.set_page_config(page_title="Credit Risk Analysis", layout="wide", initial_sidebar_state="collapsed")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Colours matching CSS
BG = "#201a12"
GOLD = "#c9a84c"
TEAL = "#00c9a7"
CRIMSON = "#c94040"
SAPPH = "#4070c9"
EMERALD = "#40a870"
INK = "#e8dcc8"
MUTED = "#a09070"

# =============================================================================
# 2. LOAD MODELS
# =============================================================================
@st.cache_resource
def fetch_models():
    data_path = "data/processed/"
    if not os.path.exists(data_path):
        return None, None, None, None, None, None, None, None, None, None
        
    try:
        with open(os.path.join(data_path, "xgb_model.pkl"), "rb") as f:
            xgb_model = pickle.load(f)
        with open(os.path.join(data_path, "xgb_calibrated.pkl"), "rb") as f:
            xgb_calibrated = pickle.load(f)
        with open(os.path.join(data_path, "cart_model.pkl"), "rb") as f:
            cart_model = pickle.load(f)
        with open(os.path.join(data_path, "best_threshold.pkl"), "rb") as f:
            best_thresh = pickle.load(f)
        comp_df = pd.read_pickle(os.path.join(data_path, "model_comparison.pkl"))
        fairness = pd.read_pickle(os.path.join(data_path, "fairness_summary.pkl"))
        master_features = pd.read_pickle(os.path.join(data_path, "master_features.pkl"))
        npa_df = pd.read_pickle(os.path.join(data_path, "rbi_npa_clean.pkl"))
        
        # New Insights Data
        lasso_coefs = pd.read_pickle(os.path.join(data_path, "lasso_coefs.pkl"))
        fairness_audit = pd.read_pickle(os.path.join(data_path, "fairness_audit_full.pkl"))
        
        return xgb_model, xgb_calibrated, cart_model, best_thresh, comp_df, fairness, master_features, npa_df, lasso_coefs, fairness_audit
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None, None, None, None, None, None

xgb_model, xgb_calibrated, cart_model, best_thresh, comp_df, fairness, master_features, npa_df, lasso_coefs, fairness_audit = fetch_models()

model_features = [
    "debt_to_income","loan_to_income","repayment_years",
    "late_payment_rate","payment_completeness","active_loan_ratio",
    "external_debt_burden","age_years","employed_years",
    "doc_submission_score","contact_score",
    "is_cash_loan","is_female","owns_car","owns_realty",
    "higher_education","prime_working_age"
]

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
LABELS = {
    "is_female": "Gender (Female)",
    "higher_education": "Higher Education Degree",
    "prime_working_age": "Prime Working Age (30-55)",
    "owns_realty": "Property Ownership",
    "owns_car": "Car Ownership",
    "is_cash_loan": "Loan Type (Cash)",
    "age_years (Quantile)": "Age (Seniority Bias)",
    "debt_to_income (Quantile)": "Debt-to-Income Pressure",
    "employed_years (Quantile)": "Career Stability",
    "late_payment_rate (Quantile)": "History Severity"
}

def build_features(income, loan_amt, annuity, goods_price, age, emp_years, gender, education,
                   loan_type, owns_car_v, owns_realty_v, late_rate, active_ratio, doc_score, contact_v, loan_mult=1.0):
    la = loan_amt * loan_mult
    vals = [
        annuity / (income + 1),
        la / (income + 1),
        la / (annuity + 1),
        late_rate, 1, active_ratio, 0, age, emp_years,
        doc_score, contact_v, loan_type, gender, owns_car_v, owns_realty_v,
        education, 1 if 30 <= age <= 55 else 0
    ]
    return pd.DataFrame([vals], columns=model_features)

def create_glow_card(title, content_html):
    # Use single-line/no-indent HTML to prevent Streamlit markdown code-block triggers
    return f'<div class="glow-card"><div class="glow-card-title">{title}</div>{content_html}</div>'

# =============================================================================
# 4. STATE-BASED BOOK NAVIGATION
# =============================================================================
if 'page' not in st.session_state:
    st.session_state.page = 0

pages = [
    "Cover", "I. Patterns", "II. Models", "III. Explanations", 
    "IV. Risk Oracle", "V. Forecast", "VI. Fairness", 
    "VII. Profit Optimizer", "VIII. Stress Test"
]
selection = pages[st.session_state.page]

st.markdown("<br>", unsafe_allow_html=True)
# Outline / Breadcrumb Trace
if selection != "Cover":
    nav_links = []
    for p in pages[1:]:
        if p == selection:
            nav_links.append(f"<b style='color: #c9a84c;'>{p.split('.')[0]}</b>")
        else:
            nav_links.append(p.split('.')[0])
    
    st.markdown(f"<div style='text-align: center; color: #a09070; font-size: 1.1em; letter-spacing: 2px; border-bottom: 1px solid rgba(201,168,76,0.2); padding-bottom: 15px;'>{ ' &nbsp; | &nbsp; '.join(nav_links) }</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# =============================================================================
# 5. PAGES
# =============================================================================

if selection == "Cover":
    st.markdown("<h1 style='text-align: center; color: #c9a84c; font-size: 5rem; margin-top: 10vh;'>Credit Risk Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #e8dcc8;'>Unveiling the hidden mathematics of loan default across 307,511 decisions</h3>", unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("<h2 style='text-align: center; color: #e8dcc8;'>307K</h2><p style='text-align: center; color: #a09070'>Applicants</p>", unsafe_allow_html=True)
    col2.markdown("<h2 style='text-align: center; color: #c94040;'>8.1%</h2><p style='text-align: center; color: #a09070'>Default Rate</p>", unsafe_allow_html=True)
    col3.markdown("<h2 style='text-align: center; color: #00c9a7;'>17</h2><p style='text-align: center; color: #a09070'>Features</p>", unsafe_allow_html=True)
    col4.markdown("<h2 style='text-align: center; color: #4070c9;'>8</h2><p style='text-align: center; color: #a09070'>Panels</p>", unsafe_allow_html=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    _, c, _ = st.columns([3,2,3])
    with c:
        if st.button("Open the Dashboard 📖", use_container_width=True):
            st.session_state.page = 1
            st.rerun()

elif selection == "I. Patterns":
    st.markdown("<h1> The Patterns of Default</h1>", unsafe_allow_html=True)
    st.markdown('<div class="narrative-box">Among 307,511 applicants, only 8.1% defaulted — yet that 8% conceals divergent results. Younger borrowers default at twice the rate of those in their forties. Late payment history is the single loudest signal the data speaks.</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="glow-card-title">Default Rate by Age Group</div>', unsafe_allow_html=True)
        df_age = pd.DataFrame({
            'Age Group': ["20-25","25-30","30-35","35-40","40-45","45-50","50-55","55-60","60-70"],
            'Default Rate (%)': [12.1,11.3,9.8,8.4,7.2,6.5,5.8,5.2,4.8]
        })
        fig = px.area(df_age, x='Age Group', y='Default Rate (%)', markers=True, color_discrete_sequence=[GOLD])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("💡 Insight: Age & Default Risk"):
            st.markdown("Younger applicants (under 30) carry a significantly higher default risk, nearly double the portfolio average. As age increases, credit stability generally improves, reflecting established careers and accumulated wealth.")

    with c2:
        st.markdown('<div class="glow-card-title">Borrower Persona Comparison (Radar)</div>', unsafe_allow_html=True)
        if master_features is not None:
            f_cols = ["late_payment_rate", "debt_to_income", "active_loan_ratio", "external_debt_burden"]
            df_radar = master_features.groupby('target')[f_cols].mean()
            df_radar_norm = df_radar / df_radar.max()
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatterpolar(r=df_radar_norm.loc[1].values, theta=["Late Payments", "DTI", "Active Loans", "Ext. Debt"], fill='toself', name='Avg. Defaulter', line_color=CRIMSON, fillcolor='rgba(201, 64, 64, 0.4)'))
            fig2.add_trace(go.Scatterpolar(r=df_radar_norm.loc[0].values, theta=["Late Payments", "DTI", "Active Loans", "Ext. Debt"], fill='toself', name='Avg. Safe Borrower', line_color=EMERALD, fillcolor='rgba(64, 168, 112, 0.4)'))
            fig2.update_layout(polar=dict(radialaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK, margin=dict(l=40, r=40, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Data loading...")
            
        with st.expander("💡 Insight: The Anatomy of a Defaulter"):
            st.markdown("Comparing statistical clusters reveals that defaulters consistently index higher on systemic late payments and extreme external debt loads, visibly skewing their structural profile.")

elif selection == "II. Models":
    st.markdown("<h1> The Three Instruments</h1>", unsafe_allow_html=True)
    if comp_df is None:
        st.warning("Models are still training... Please wait and reload.")
        st.stop()
        
    st.markdown('<div class="narrative-box">Three models competed for primacy. Lasso pruned the logistic model to its essential predictors. CART carved the population into readable risk segments. XGBoost synthesised every signal into its finest discriminating instrument.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    logit_auc = comp_df[comp_df["Model"].str.contains("Lasso")]["AUC"].values[0]
    cart_auc = comp_df[comp_df["Model"] == "CART (pruned)"]["AUC"].values[0]
    xgb_auc = comp_df[comp_df["Model"] == "XGBoost"]["AUC"].values[0]
    
    col1.markdown(create_glow_card("Logistic (Lasso)", f"<div class='metric-container'><div class='metric-value gold-text'>{logit_auc:.3f}</div><div class='metric-label'>AUC</div></div>"), unsafe_allow_html=True)
    col2.markdown(create_glow_card("CART (pruned)", f"<div class='metric-container'><div class='metric-value teal-text'>{cart_auc:.3f}</div><div class='metric-label'>AUC</div></div>"), unsafe_allow_html=True)
    col3.markdown(create_glow_card("XGBoost — Champion", f"<div class='metric-container'><div class='metric-value sapph-text'>{xgb_auc:.3f}</div><div class='metric-label'>AUC</div></div>"), unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="glow-card-title">Performance Comparison</div>', unsafe_allow_html=True)
        df_melt = comp_df.melt(id_vars=["Model"], value_vars=["AUC", "F1_Score", "KS_Stat", "Gini"], var_name="Metric", value_name="Score")
        fig = px.line(df_melt, x="Metric", y="Score", color="Model", markers=True, color_discrete_map={comp_df["Model"].values[0]: GOLD, "CART (pruned)": TEAL, "XGBoost": SAPPH})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("💡 Insight: Algorithm Selection"):
            st.markdown("XGBoost achieves the highest AUC, correctly ranking complex non-linear interactions. However, Logistic Regression with Lasso provides excellent baseline discriminative power while remaining 100% transparent for regulatory mandates.")
    with c2:
        st.markdown('<div class="glow-card-title">Lasso Selection Efficiency</div>', unsafe_allow_html=True)
        if lasso_coefs is not None:
            # Filter non-zero
            lasso_plot = lasso_coefs[lasso_coefs['Coefficient'] != 0].copy()
            lasso_plot['Sign'] = lasso_plot['Coefficient'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
            fig_lasso = px.bar(lasso_plot, x="Coefficient", y="Feature", orientation='h', color="Sign", color_discrete_map={"Positive": CRIMSON, "Negative": EMERALD})
            fig_lasso.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK, height=300, showlegend=False, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_lasso, use_container_width=True)
            st.markdown(f"<p style='color:{MUTED}; font-size:0.8em;'>Lasso pruned **{len(lasso_coefs) - len(lasso_plot)}** irrelevant features to zero.</p>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glow-card-title">The Rule Engine: CART Decision Tree Logic</div>', unsafe_allow_html=True)
    if cart_model is not None:
        fig_tree, ax_tree = plt.subplots(figsize=(20, 8), facecolor=BG)
        plot_tree(cart_model, max_depth=3, feature_names=model_features, class_names=["Safe", "Default"], 
                  filled=True, rounded=True, ax=ax_tree, fontsize=10, 
                  label='none', proportion=True, node_ids=False)
        st.pyplot(fig_tree)

elif selection == "III. Explanations":
    st.markdown("<h1> Deep Explanations</h1>", unsafe_allow_html=True)
    if master_features is None or xgb_model is None:
        st.warning("Models not ready.")
        st.stop()
        
    st.markdown('<div class="narrative-box">Behind each overall score lies a balance of competing models. Global SHAP reveals the results across all 300,000 customers, but we must also look into Local SHAP to understand why an individual customer is rejected.</div>', unsafe_allow_html=True)
    
    with st.spinner("Calculating Explanations..."):
        df_sample = master_features[model_features].dropna().sample(2000, random_state=42)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(df_sample)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown('<div class="glow-card-title">Integrated Feature Importance: Model Gain vs SHAP impact</div>', unsafe_allow_html=True)
            # Compare built-in importance with SHAP
            imp_df = pd.DataFrame({
                "Feature": model_features,
                "Model Gain": xgb_model.feature_importances_,
                "Impact": np.abs(shap_values).mean(axis=0)
            })
            # Normalize for side-by-side
            imp_df["Model Gain"] = imp_df["Model Gain"] / imp_df["Model Gain"].max()
            imp_df["Impact"] = imp_df["Impact"] / imp_df["Impact"].max()
            
            imp_melt = imp_df.melt(id_vars="Feature", var_name="Method", value_name="Relative Importance").sort_values("Relative Importance", ascending=True)
            fig_imp = px.bar(imp_melt, x="Relative Importance", y="Feature", color="Method", barmode='group', color_discrete_map={"Model Gain": MUTED, "Impact": SAPPH})
            fig_imp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK, height=500)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with c2:
            st.markdown('<div class="glow-card-title">Global View: Feature Dependence Dynamics</div>', unsafe_allow_html=True)
            def plot_shap_grid(f_name, col):
                idx = model_features.index(f_name)
                df_dep = pd.DataFrame({f_name: df_sample[f_name], "SHAP": shap_values[:, idx]})
                fig = px.scatter(df_dep, x=f_name, y="SHAP", color="SHAP", color_continuous_scale=[SAPPH, MUTED, CRIMSON])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK, height=220, margin=dict(t=5, b=5), showlegend=False)
                col.plotly_chart(fig, use_container_width=True)
            
            sc1, sc2 = st.columns(2)
            plot_shap_grid("late_payment_rate", sc1)
            plot_shap_grid("debt_to_income", sc2)
            sc3, sc4 = st.columns(2)
            plot_shap_grid("age_years", sc3)
            plot_shap_grid("active_loan_ratio", sc4)

        st.markdown("<hr>", unsafe_allow_html=True)
        
 #       c_l, c_r = st.columns([1, 2])
  #      with c_l:
   #         st.markdown('<div class="glow-card-title">Local Insight: Individual Waterfall</div>', unsafe_allow_html=True)
    #        user_idx = 42 
   #         base_val = explainer.expected_value
    #        if isinstance(base_val, np.ndarray): base_val = base_val[0]
     #       user_shaps = shap_values[user_idx]
      #      user_data = df_sample.iloc[user_idx]
    #        shap_df = pd.DataFrame({"Feature": model_features, "Value": user_data.values, "SHAP": user_shaps})
     #       shap_df = shap_df.reindex(shap_df.SHAP.abs().sort_values(ascending=False).index).head(6) 
            
      #      with st.expander("💡 Insight: The Human Story", expanded=True):
       #         st.markdown(f"**Applicant #{user_idx} Summary**\nThis borrower is pushed into the 'Dangerous' red zone primarily by extreme **Late Payment History** (+{shap_df[shap_df['Feature'] == 'late_payment_rate']['SHAP'].values[0]:.2f} SHAP penalty). Even though their Age provides a minor protective buffer (green), it is completely insufficient to rescue the application.")
            
     #   with c_r:
    #        fig_water = go.Figure(go.Waterfall(
     #           orientation = "h", measure = ["relative"] * 6 + ["total"],
      #          y = shap_df['Feature'].tolist() + ["Final Prediction Log-Odds"],
    #            x = shap_df['SHAP'].tolist() + [base_val + shap_df['SHAP'].sum()],
     #           connector = {"line": {"color": MUTED, "width": 2, "dash":"dot"}},
      #          decreasing = {"marker": {"color": EMERALD}}, increasing = {"marker": {"color": CRIMSON}},
    #            totals = {"marker": {"color": GOLD}}
     #       ))
      #      fig_water.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK, title=f"Risk Breakdown for Applicant #{user_idx}", height=400)
      #      st.plotly_chart(fig_water, use_container_width=True)'''

elif selection == "IV. Risk Oracle":
    st.markdown("<h1> The Risk Oracle</h1>", unsafe_allow_html=True)
    if xgb_model is None:
        st.warning("Models not ready.")
        st.stop()
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown('<div class="glow-card-title">Financial Profile</div>', unsafe_allow_html=True)
        income = st.number_input("Annual income (INR)", value=200000, step=10000)
        loan = st.number_input("Loan amount (INR)", value=500000, step=10000)
        annuity = st.number_input("Annuity (INR/yr)", value=30000, step=1000)
        goods = st.number_input("Goods price (INR)", value=450000, step=10000)
        st.markdown('<div class="glow-card-title">Personal</div>', unsafe_allow_html=True)
        age = st.number_input("Age", value=35, min_value=18, max_value=70)
        emp = st.number_input("Employment (yrs)", value=5, min_value=0)
        gender = st.selectbox("Gender", options=[("Female", 1), ("Male", 0)], format_func=lambda x: x[0])[1]
        edu = st.selectbox("Education", options=[("Higher", 1), ("Other", 0)], format_func=lambda x: x[0])[1]
        st.markdown('<div class="glow-card-title">History & Signals</div>', unsafe_allow_html=True)
        late = st.slider("Late payment rate", 0.0, 1.0, 0.05)
        active = st.slider("Active loan ratio", 0.0, 1.0, 0.3)
        doc = st.slider("Docs submitted", 0, 20, 3)
        contact = st.slider("Contact channels", 0, 5, 3)

    with c2:
        feats_df = build_features(income, loan, annuity, goods, age, emp, gender, edu, 1, 0, 0, late, active, doc, contact)
        raw_prob = xgb_model.predict_proba(feats_df)[:, 1]
        prob = xgb_calibrated.predict(raw_prob)[0]
        is_high_risk = prob > best_thresh
        orb_class = "high-risk" if is_high_risk else "low-risk"
        decision = f"Refer for Review (thresh={best_thresh:.3f})" if is_high_risk else f"Approve (thresh={best_thresh:.3f})"
        st.markdown(f"""<div class="orb-container glow-card"><div class="orb {orb_class}">{prob*100:.1f}%</div><div style="margin-top: 20px; font-size: 1.2rem;font-weight:bold;">{decision}</div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="glow-card-title">Business Impact: Approval Funnel Pipeline</div>', unsafe_allow_html=True)
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=["All Apps", "Clean History", "Late Payments", "Safe (Approve)", "Flagged (Review)", "Reject"], color=[SAPPH, TEAL, CRIMSON, EMERALD, GOLD, CRIMSON]),
            link=dict(source=[0, 0, 1, 1, 2, 2], target=[1, 2, 3, 4, 4, 5], value=[70, 30, 65, 5, 10, 20], color=['rgba(0, 201, 167, 0.2)', 'rgba(201, 64, 64, 0.2)', 'rgba(64, 168, 112, 0.3)', 'rgba(201, 168, 76, 0.3)', 'rgba(201, 168, 76, 0.3)', 'rgba(201, 64, 64, 0.3)'])
        )])
        fig_sankey.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK, height=300, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_sankey, use_container_width=True)

elif selection == "V. Forecast":
    st.markdown("<h1> The NPA Chronicle</h1>", unsafe_allow_html=True)
    if npa_df is None:
        st.warning("NPA Data missing.")
        st.stop()
    horizon = st.slider("Forecast Horizon (Years)", 1, 10, 5)
    model = ARIMA(npa_df['npa_ratio'].values, order=(1,1,0))
    fit_model = model.fit()
    forecast = fit_model.get_forecast(steps=horizon)
    fc_mean = forecast.predicted_mean
    np.random.seed(42); n_sim = 1000; resid = fit_model.resid; simulations = np.zeros((horizon, n_sim))
    for i in range(n_sim):
        for t in range(horizon): simulations[t, i] = fc_mean[t] + np.random.choice(resid)
    l5, l25, med, u75, u95 = np.percentile(simulations, [5, 25, 50, 75, 95], axis=1)
    last_year = int(npa_df['year'].max()); future_years = np.arange(last_year+1, last_year+1+horizon)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=npa_df['year'], y=npa_df['npa_ratio'], mode='lines+markers', name='Historical', line=dict(color=SAPPH, width=3)))
    fig.add_trace(go.Scatter(x=np.concatenate([future_years, future_years[::-1]]), y=np.concatenate([u95, l5[::-1]]), fill='toself', fillcolor='rgba(201, 64, 64, 0.15)', line=dict(color='rgba(255,255,255,0)'), showlegend=True, name='5th-95th Percentile'))
    fig.add_trace(go.Scatter(x=np.concatenate([future_years, future_years[::-1]]), y=np.concatenate([u75, l25[::-1]]), fill='toself', fillcolor='rgba(201, 64, 64, 0.3)', line=dict(color='rgba(255,255,255,0)'), showlegend=True, name='Interquartile Range (IQR)'))
    fig.add_trace(go.Scatter(x=future_years, y=med, mode='lines', name='Base Case', line=dict(color=CRIMSON, dash='dash', width=2)))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK, height=500, xaxis_title="Year", yaxis_title="NPA Ratio (%)")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("💡 Insight: The Horizon of Risk"):
        st.markdown(f"The ARIMA model (1,1,0) captures the momentum of historical banking NPAs. The widening ribbons represent the **Cone of Uncertainty**—while the base case suggests relative stability, the outer boundaries warn of a 5% possibility where macro shocks push the default rate significantly higher over the next {horizon} years.")

elif selection == "VI. Fairness":
    st.markdown("<h1> The Global Bias Auditor</h1>", unsafe_allow_html=True)
    if fairness_audit is None:
        st.warning("Audit Data missing.")
        st.stop()
    
    st.markdown('<div class="narrative-box">We transitioned from single-variable checks to a Full Portfolio Bias Auditor. This system automatically identifies Disparate Impact across every binary and continuous attribute in the model, identifying where systemic imbalance resides.</div>', unsafe_allow_html=True)
    
    # 1. Main Global Auditor Chart
    st.markdown('<div class="glow-card-title">Portfolio-Wide Disparate Impact Ratio (DIR)</div>', unsafe_allow_html=True)
    
    # Logic to color-code based on 80% rule
    fairness_audit['Status'] = fairness_audit['DIR'].apply(lambda x: 'Violation' if x < 0.8 else 'Ideal' if x >= 0.95 else 'Concern')
    fig_bias = px.bar(fairness_audit, x="DIR", y="Attribute", orientation='h', color="Status", color_discrete_map={'Violation': CRIMSON, 'Concern': GOLD, 'Ideal': EMERALD})
    fig_bias.add_vline(x=0.8, line_dash="dash", line_color=GOLD, annotation_text="80% Threshold")
    fig_bias.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK, height=500, xaxis_title="Disparate Impact Ratio (DIR)")
    st.plotly_chart(fig_bias, use_container_width=True)
    
    violations = fairness_audit[fairness_audit['DIR'] < 0.8]['Attribute'].tolist()
    
    col_l, col_r = st.columns([1, 2])
    with col_l:
        risk_items_html = ""
        for v in violations:
            label = LABELS.get(v, v)
            # Constructed as a single line to avoid markdown indentation bugs
            risk_items_html += f"<div style='margin-bottom:12px;padding:8px;background:rgba(201,64,64,0.1);border-left:3px solid {CRIMSON};border-radius:4px;'><div style='font-weight:bold;color:{INK};'>{label}</div><div style='font-size:0.8em;color:{MUTED};'>Unprivileged group approval is significantly lower.</div></div>"
        
        st.markdown(create_glow_card("At-Risk Attributes", f"<div style='color:{CRIMSON};font-weight:bold;margin-bottom:15px;'>{len(violations)} Potential Biases Identified</div>{risk_items_html}"), unsafe_allow_html=True)
        
        st.markdown(f"<div style='font-size:0.85em;color:{MUTED};padding:10px;border:1px solid rgba(255,255,255,0.1);border-radius:5px;'><b>Action:</b> Consider applying <b>Reweighting</b> or <b>Equal Odds</b> constraints on these features in Chapter VII if these biases are social (Gender/Age) rather than economic.</div>", unsafe_allow_html=True)
        
    with col_r:
        st.markdown('<div class="glow-card-title">Bias Spotlight Explorer</div>', unsafe_allow_html=True)
        # Selectbox with human names
        audit_options = fairness_audit['Attribute'].tolist()
        sel_attr = st.selectbox("Select Attribute for Deep Dive", options=audit_options, format_func=lambda x: LABELS.get(x, x))
        
        target_dir = fairness_audit[fairness_audit['Attribute'] == sel_attr]['DIR'].values[0]
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.markdown(f"**Disparate Impact Ratio:**")
            color = CRIMSON if target_dir < 0.8 else EMERALD
            st.markdown(f"<h2 style='color:{color}; margin-top:0;'>{target_dir:.3f}</h2>", unsafe_allow_html=True)
        with c_m2:
            status = "VIOLATION" if target_dir < 0.8 else "PASS"
            st.markdown(f"**Status:**")
            st.markdown(f"<h3 style='color:{color}; margin-top:5px;'>{status}</h3>", unsafe_allow_html=True)

        with st.expander("💡 Insight: Root Cause Analysis", expanded=True):
            if target_dir < 0.8:
                st.markdown(f"The model significantly favors the privileged group for **{sel_attr}**. This may be due to historical training bias or strong proxy-correlations with income levels that are unevenly distributed across the population.")
            else:
                st.markdown(f"The model maintains mathematical parity for **{sel_attr}**, ensuring that approval decisions are made independently of this specific attribute's membership.")

elif selection == "VII. Profit Optimizer":
    st.markdown("<h1> The Policy & Profit Optimizer</h1>", unsafe_allow_html=True)
    if master_features is None: st.warning("Models not ready."); st.stop()
    rev_per_good = st.number_input("Est. Revenue per Safe Loan (INR)", value=50000, step=5000)
    loss_per_bad = st.number_input("Est. Loss per Default (INR)", value=-250000, step=25000)
    calc_df = master_features[['target']].dropna().sample(min(15000, len(master_features)), random_state=42)
    t_labels = calc_df['target'].values
    cX = master_features.loc[calc_df.index][model_features]; raw_p = xgb_model.predict_proba(cX)[:, 1]
    probs = xgb_calibrated.predict(raw_p); threshs = np.linspace(0.01, 0.5, 50); profits = []
    for t in threshs: preds = (probs >= t).astype(int); profit = np.sum((preds == 0) & (t_labels == 0)) * rev_per_good + np.sum((preds == 0) & (t_labels == 1)) * loss_per_bad; profits.append(profit)
    df_prof = pd.DataFrame({"Threshold": threshs, "Total Profit": profits}); opt_t = threshs[np.argmax(profits)]; max_p = np.max(profits)
    fig = px.area(df_prof, x="Threshold", y="Total Profit", color_discrete_sequence=[EMERALD])
    fig.add_vline(x=opt_t, line_dash="dash", line_color=GOLD, annotation_text=f"Optimal Thr: {opt_t:.3f}")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=INK)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("💡 Insight: Maximizing Yield vs. Risk"):
        st.markdown(f"The optimal threshold of **{opt_t:.3f}** is the mathematical 'sweet spot' where the bank's revenue from safe loans perfectly offsets the cost of predicted defaults. Moving the threshold to the left increases volume but risks capital; moving it to right protects capital but shrinks the customer base.")

elif selection == "VIII. Stress Test":
    st.markdown("<h1> Macroeconomic Stress Testing</h1>", unsafe_allow_html=True)
    if master_features is None: st.warning("Models not ready."); st.stop()
    dti_shock = st.slider("DTI Shock (Inflation/Cost of Living)", 1.0, 2.0, 1.25, 0.05)
    late_shock = st.slider("Late Payments Shock (Systemic Freezes)", 1.0, 3.0, 1.5, 0.1)
    with st.spinner("Simulating..."):
        base_df = master_features[model_features].dropna().sample(min(10000, len(master_features)), random_state=42)
        base_raw = xgb_model.predict_proba(base_df)[:, 1]; base_p = xgb_calibrated.predict(base_raw)
        base_def = np.sum(base_p >= best_thresh); base_dr = base_def / len(base_df)
        shocked_df = base_df.copy(); shocked_df['debt_to_income'] *= dti_shock; shocked_df['late_payment_rate'] *= late_shock
        drift = dti_shock - 1.0; shocked_df['external_debt_burden'] *= (1.0 + (drift * 0.5)); shocked_df['loan_to_income'] *= (1.0 + (drift * 0.8))
        s_raw = xgb_model.predict_proba(shocked_df)[:, 1]; s_p = xgb_calibrated.predict(s_raw)
        s_def = np.sum(s_p >= best_thresh); s_dr = s_def / len(base_df)
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Baseline Default Rate", value=f"{base_dr*100:.1f}%")
        c2.metric(label="Shocked Default Rate", value=f"{s_dr*100:.1f}%", delta=f"+{(s_dr-base_dr)*100:.1f}%", delta_color="inverse")
        c3.metric(label="Added Capital at Risk", value=f"₹{(s_def-base_def)*500000:,.0f}", delta="Severe Threat", delta_color="inverse")
    
    with st.expander("💡 Insight: The Cost of Turbulence"):
        st.markdown("Macroeconomic shocks are rarely isolated. By simulating **Covariant Drift**, we see that a surge in inflation doesn't just increase DTI; it naturally drags up external debt burdens and late payment frequencies across the entire portfolio, rapidly eroding the bank's capital buffers.")

# =============================================================================
# 6. BOOK FLIP CONTROLS
# =============================================================================
st.markdown("<br><hr><br>", unsafe_allow_html=True)
if selection != "Cover":
    bot_c1, bot_c2, bot_c3 = st.columns([1, 8, 1])
    with bot_c1:
        if st.session_state.page > 0:
            if st.button("◀ Prev"): st.session_state.page -= 1; st.rerun()
    with bot_c3:
        if st.session_state.page < len(pages) - 1:
            if st.button("Next ▶"): st.session_state.page += 1; st.rerun()
    