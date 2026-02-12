"""Health Claims Fact-Checker - Streamlit UI."""

import streamlit as st

st.set_page_config(
    page_title="Health Claims Fact-Checker",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Health Claims Fact-Checker")
st.markdown("Verify health claims against peer-reviewed research, clinical trials, and medical guidelines.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    system_variant = st.selectbox(
        "System Variant",
        ["S5: Full Pipeline", "S4: Multi-method", "B3: API + Vector DB", "B2: API-only", "B1: No retrieval"],
    )
    
    st.divider()
    
    st.subheader("Retrieval Methods")
    use_pubmed = st.checkbox("PubMed API", value=True)
    use_scholar = st.checkbox("Semantic Scholar", value=True)
    use_cochrane = st.checkbox("Cochrane", value=True)
    use_trials = st.checkbox("ClinicalTrials.gov", value=True)
    use_guidelines = st.checkbox("Guideline Store", value=True)
    use_deep_search = st.checkbox("Full-text Deep Search", value=True)
    use_vlm = st.checkbox("VLM Figure Extraction", value=True)
    
    st.divider()
    
    show_trace = st.checkbox("Show Agent Trace", value=True)
    show_cost = st.checkbox("Show Cost Breakdown", value=True)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    claim = st.text_area(
        "Enter a health claim to verify:",
        placeholder="e.g., Intermittent fasting reverses Type 2 diabetes",
        height=100,
    )

with col2:
    st.markdown("**Try an example:**")
    examples = [
        "Intermittent fasting reverses Type 2 diabetes",
        "Vitamin D prevents COVID-19",
        "Vaccines cause autism",
        "Turmeric cures cancer",
        "Aspirin is a blood thinner",
    ]
    for ex in examples:
        if st.button(ex[:30] + "...", key=ex):
            claim = ex

if st.button("🔍 Verify Claim", type="primary"):
    if not claim:
        st.warning("Please enter a claim to verify.")
    else:
        with st.spinner("Verifying claim..."):
            # TODO: Call the actual pipeline
            st.info("Pipeline not yet implemented. This is a placeholder UI.")
            
            # Placeholder verdict
            st.success("## ⚠️ OVERSTATED")
            st.metric("Confidence", "78%")
            st.markdown("""
            **Explanation:** While intermittent fasting shows promise for glycemic control,
            calling it a "reversal" overstates the evidence. Most studies show improvement,
            not remission. Long-term evidence is limited.
            """)

# Tabs for details
tab1, tab2, tab3, tab4 = st.tabs(["Agent Trace", "Evidence", "Figures", "Retrieval Methods"])

with tab1:
    st.markdown("### Agent Execution Trace")
    st.info("Agent trace will appear here after verification.")

with tab2:
    st.markdown("### Retrieved Evidence")
    st.info("Evidence will appear here after verification.")

with tab3:
    st.markdown("### Extracted Figures")
    st.info("VLM-extracted figures will appear here after verification.")

with tab4:
    st.markdown("### Retrieval Methods Used")
    st.info("Retrieval method breakdown will appear here after verification.")
