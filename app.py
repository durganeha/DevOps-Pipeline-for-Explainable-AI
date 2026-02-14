import streamlit as st
import time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import src.fairness_check as fairness_check

# --- Page Configuration ---
st.set_page_config(page_title="Judicial AI Audit Pipeline", page_icon="⚖️", layout="wide")

# --- Initialize Session State ---
if 'pipeline_executed' not in st.session_state:
    st.session_state.pipeline_executed = False
if 'audit_results' not in st.session_state:
    st.session_state.audit_results = {}

# --- Header ---
st.title("⚖️ MLOps Pipeline: Fairness & XAI Gate")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("🔧 Developer Controls")
    uploaded_file = st.file_uploader("Upload Model Artifact (.pkl)", type=["pkl"])
    st.divider()
    threshold = st.slider("Min. Disparate Impact Threshold", 0.0, 1.0, 0.8)
    
    if st.button("♻️ Reset Pipeline"):
        st.session_state.pipeline_executed = False
        st.session_state.audit_results = {}
        st.rerun()

# --- Main Logic ---
if uploaded_file is None:
    st.info("👋 Please upload a model artifact to begin.")
else:
    data = pickle.load(uploaded_file)
    model = data['model']
    scaler = data['scaler']
    X_test, y_true = data['test_data']
    feature_names = data['feature_names']

    if st.button("▶️ EXECUTE PIPELINE"):
        with st.status("Running Pipeline Stages...", expanded=True) as status:
            st.write("🧪 Preparing Data...")
            X_test_scaled = scaler.transform(X_test)
            
            st.write("⚖️ Auditing Fairness...")
            y_pred = model.predict(X_test_scaled)
            results = fairness_check.run_audit(X_test, y_pred, y_true)
            
            st.write("🔍 Calculating SHAP Values...")
            explainer = shap.LinearExplainer(model, X_test_scaled)
            shap_values = explainer.shap_values(X_test_scaled)
            
            st.session_state.audit_results = {
                'di_score': results['DI'],
                'shap_values': shap_values,
                'explainer': explainer,
                'X_test_scaled': X_test_scaled
            }
            st.session_state.pipeline_executed = True
            status.update(label="✅ Pipeline Complete", state="complete")

    if st.session_state.pipeline_executed:
        res = st.session_state.audit_results
        
        st.header("📊 Automated Compliance Report")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fairness Audit")
            di = res['di_score']
            st.metric("Disparate Impact Score", f"{di:.2f}")
            if di >= threshold:
                st.success("✅ PASSED: Model satisfies fairness rule.")
            else:
                st.error("❌ FAILED: Significant racial bias detected.")

        with col2:
            st.subheader("Global Explainability (XAI)")
            fig_sum, ax_sum = plt.subplots()
            shap.summary_plot(res['shap_values'], res['X_test_scaled'], feature_names=feature_names, show=False)
            st.pyplot(fig_sum)

        st.divider()
        st.subheader("🧪 Manual 'What-If' Audit")
        
        with st.form("manual_audit_form"):
            ca, cb, cc, cd = st.columns(4)
            with ca: m_age = st.number_input("Age", 18, 100, 25)
            with cb: m_priors = st.number_input("Prior Crimes", 0, 50, 2)
            with cc: m_race = st.selectbox("Race", [0, 1], format_func=lambda x: "Unprivileged (0)" if x==0 else "Privileged (1)")
            with cd: m_emp = st.selectbox("Employment", [0, 1], format_func=lambda x: "Unemployed (0)" if x==0 else "Employed (1)")
            
            submit_manual = st.form_submit_button("Run Individual Audit")

        if submit_manual:
            # 1. Prepare and scale the data
            m_raw = pd.DataFrame([[m_age, m_priors, m_race, m_emp]], columns=feature_names)
            m_scaled = scaler.transform(m_raw)
            
            # 2. Generate Prediction and standardized SHAP variable name
            m_shap_values = res['explainer'].shap_values(m_scaled)
            
            st.write("### Result for this entry:")
            fig_m, ax_m = plt.subplots()
            m_exp = shap.Explanation(
                values=m_shap_values[0], 
                base_values=res['explainer'].expected_value, 
                data=m_raw.iloc[0], 
                feature_names=feature_names
            )
            shap.plots.waterfall(m_exp, show=False)
            st.pyplot(fig_m)

            # 3. Human-Readable Summary
            st.write("### 📝 Summary")
            explanations = []
            
            for i, feature in enumerate(feature_names):
                shap_val = m_shap_values[0][i]  # Corrected variable name
                feature_val = m_raw.iloc[0][i]  # Corrected to use manual input data
                
                if abs(shap_val) < 0.1:
                    continue 

                direction = "increased" if shap_val > 0 else "decreased"
                impact = "higher" if shap_val > 0 else "lower"

                if feature == 'race':
                    val_str = "Unprivileged group" if feature_val == 0 else "Privileged group"
                elif feature == 'employment':
                    val_str = "Unemployed" if feature_val == 0 else "Employed"
                else:
                    val_str = str(feature_val)

                explanations.append(f"• The **{feature}** ({val_str}) **{direction}** the risk score, making the individual appear **{impact} risk** to the model.")

            if explanations:
                for line in explanations:
                    st.write(line)
            else:
                st.write("All features had a neutral impact on this specific decision.")