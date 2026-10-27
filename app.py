
import os
import pandas as pd
import streamlit as st

from engine import compare, policies

st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="📋",
    layout="wide"
)

# Load Streamlit secrets
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

st.title("Company Policy Assistant")

st.caption(
    "Comparison of rules-based search, "
    "full-context LLM and vector-index retrieval"
)

st.write(f"Policy database: {len(policies())} records")

question = st.text_input(
    "Ask a policy question",
    value="What is the dress code?"
)

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Compare approaches", type="primary"):
    if question.strip():
        with st.spinner("Comparing approaches..."):
            st.session_state.results = compare(question)

if st.session_state.results is not None:
    results = st.session_state.results
    df = pd.DataFrame(results)

    st.subheader("Comparison results")

    # Compact table with performance metrics
    columns = [
        "Approach",
        "Response time (s)",
        "Total tokens",
        "Input tokens",
        "Output tokens",
        "Status"
    ]

    st.dataframe(
        df[columns],
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Full answers")

    for row in results:
        with st.container(border=True):
            st.markdown(f"### {row['Approach']}")

            st.write(row["Answer"])

            st.markdown("**Relevant policy:**")
            st.write(row["Relevant policy"])

    st.info(
        "Rules-based search returns policy text directly. "
        "LLM answers may contain unsupported statements. "
        "The vector-index approach uses Gemini Embedding "
        "to retrieve relevant policies before generating "
        "an answer. Retrieved policies are candidates, "
        "not automatically verified citations."
    )

    st.download_button(
        label="Download results CSV",
        data=df.to_csv(index=False),
        file_name="policy_comparison.csv",
        mime="text/csv"
    )
