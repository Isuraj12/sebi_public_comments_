import pandas as pd 
import numpy as np 
import streamlit as st 
import google.generativeai as genai 
import sqlite3 
import os

# --- Configure Gemini ---

genai.configure(api_key= "AIzaSyCQo0iMnakG1dulc-nQbJNUWB25JdNuqew")
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Function to Load Data ---
def get_data():
    conn = sqlite3.connect('sebi_circulars.db')
    df = pd.read_sql('SELECT * FROM circulars', conn)
    conn.close()
    return df

# --- Helper to parse mixed date formats ---
def parse_dates(series):
    # Try multiple known formats (SEBI website uses various)
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
        "%b %d, %Y", "%B %d, %Y"
    ]
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed
    # fallback: try generic fuzzy parsing
    return pd.to_datetime(series, errors="coerce", dayfirst=True)

# --- Streamlit App ---
st.title(" SEBI Draft Circular Analysis")

df = get_data()

if df.empty:
    st.warning("No data available.")
else:
    if "Date" not in df.columns or "Title" not in df.columns:
        st.error("The 'Date' or 'Title' column is missing from the data.")
    else:
        # --- Robust Date Conversion ---
        df["Date"] = parse_dates(df["Date"])
        valid_dates = df["Date"].dropna()

        if valid_dates.empty:
            st.error(" Could not parse any valid dates. Check your 'Date' column format in the database.")
            st.dataframe(df.head())  # show a preview for debugging
        else:
            # --- Filter Section ---
            st.sidebar.header(" Filter Circulars")

            # Title filter
            title_filter = st.sidebar.text_input("Search by Title (keywords):", "")

            # Safe handling of min/max dates
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            # Date range filter
            start_date, end_date = st.sidebar.date_input(
                "Filter by Date Range:",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )

            # Apply filters
            filtered_df = df.copy()

            if title_filter:
                filtered_df = filtered_df[
                    filtered_df["Title"].str.contains(title_filter, case=False, na=False)
                ]

            if isinstance(start_date, list) or isinstance(start_date, tuple):
                start_date, end_date = start_date[0], start_date[1]

            filtered_df = filtered_df[
                (filtered_df["Date"] >= pd.to_datetime(start_date)) &
                (filtered_df["Date"] <= pd.to_datetime(end_date))
            ]

            # Display filtered options
            if filtered_df.empty:
                st.warning("No circulars found for the selected filters.")
            else:
                filtered_df["Display"] = filtered_df["Date"].dt.strftime("%d %b %Y") + " - " + filtered_df["Title"].astype(str)

                selected_circular = st.selectbox("Select a Circular:", filtered_df["Display"].tolist())

                if selected_circular:
                    selected_row = filtered_df[filtered_df["Display"] == selected_circular].iloc[0]

                    st.subheader("Circular Details")
                    if pd.notna(selected_row['Date']):
                        st.write(f"**Date:** {selected_row['Date'].strftime('%d %b %Y')}")
                    else:
                        st.write("**Date:** Not Available")
                    st.write(f"**Title:** {selected_row['Title']}")

                    if "PDF_URL" in df.columns and pd.notna(selected_row["PDF_URL"]):
                        st.markdown(f"[View PDF]({selected_row['PDF_URL']})")

                    if "Extracted_Text" in df.columns:
                        circular_text = selected_row["Extracted_Text"]

                        st.subheader("Extracted Text")
                        st.text_area("Extracted Text:", circular_text, height=300)

                        
                        if st.button(" Generate Summary"):
                            with st.spinner("Generating Summary..."):
                                prompt = f"""
You are a Financial Regulatory Analyst with expertise in SEBI regulations, securities law, and capital-market policy.

The user will provide the extracted text of a SEBI Consultation Paper or Draft Circular.

Your task is to produce a structured and comprehensive review document in the format used by legal-policy and market-regulation think tanks, maintaining clarity, neutrality, and analytical depth.

The review must handle any number of proposals — from one to many — without truncation or omission.
If the Consultation Paper contains several proposals, analyze and document each proposal separately and completely.
If sections are repetitive or similar, summarize only once where appropriate but do not skip or merge distinct proposals.

🔹 OUTPUT FORMAT
Market Classification

At the top, classify the Consultation Paper into one of the following:

Primary Markets – IPOs, FPOs, REITs, InvITs, public issues, capital formation

Secondary Markets – Trading, exchanges, intermediaries, surveillance, disclosures, investor protection

Commodity Markets – Commodity exchanges and derivatives

External Markets – Cross-border listings, FPIs, GDRs, ADRs, or external capital flows

1. Background / Regulatory Context / Introduction

Include:

A clear overview of the purpose and subject of the Consultation Paper.

The existing SEBI regulatory framework (Regulations, Master Circulars, previous circulars).

The evolution and pain points that led to this reform.

Any historical or legislative references (e.g., previous amendments, dates).

A short purpose statement: one or two sentences on the aim of the proposed reform.

2. Summary of Key Proposals

Provide a neutral, structured summary of all proposals in the Consultation Paper.
Each proposal should be clearly enumerated:

Example format:

Proposal 1: Title or Key Topic

Summary of the proposed regulatory change.

Specific aspects (scope, entities covered, timelines, thresholds, etc.)

Proposal 2: Title or Key Topic

Summary of what SEBI proposes.

(Continue sequentially for all proposals — do not omit or group multiple ones unless explicitly identical.)

Avoid commentary here; this section must remain purely descriptive.

3. Critical Analysis of the Proposals

For each proposal, create a distinct, detailed sub-section with the following structure.
If there are many proposals, repeat this structure as Proposal 1, Proposal 2, Proposal 3, ... Proposal N — continue for all without limit.

Proposal [Number]: [Title of Proposal]

Concept Proposed:
Provide a concise summary of the specific reform proposal, as written in the Consultation Paper.

SEBI’s Rationale:
Explain SEBI’s reasoning or objectives — what market inefficiency or regulatory gap it addresses.

Global Benchmarking:
Compare SEBI’s approach with corresponding practices in 3–4 key jurisdictions (choose from: US SEC, UK FCA, EU ESMA, Singapore MAS, Hong Kong SFC/HKEX).
Explain similarities, divergences, and global best practices.
Provide URLs or policy references if available.

Critical Assessment & Recommendations:

Our Stance:
Choose one of the following:

Accepted

Accepted with Modifications

Not Accepted

Supporting Rationale:
Provide an analytical discussion of why this stance is taken — considering implications for regulation, compliance, investor protection, and market integrity.

Proposed Modifications / Safeguards (if applicable):
Suggest specific, actionable changes or enhancements such as:

Revised thresholds or quantitative criteria

Additional disclosure or audit safeguards

Phased implementation or transitional provisions

Clarifications to avoid duplication or regulatory overlap

4. Conclusion and Overall Recommendations

Summarize the overall findings and recommendations across all proposals.
Include:

Whether SEBI’s overall approach is conceptually sound and globally aligned.

Likely impacts on market efficiency, investor confidence, and compliance burden.

A concise list of 3–5 recommendations on how SEBI could improve or clarify its proposals before finalizing the framework.

Use bullet points for clarity.

5. Key Questions for the Ministry of Finance (MoF)

Frame five policy-level questions the Ministry should ask SEBI regarding this Consultation Paper.
Questions should challenge assumptions, implementation logic, or strategic alignment.

Example structure:

How will SEBI ensure that [specific reform] avoids duplicative compliance obligations for listed entities?

What impact assessment methodology has SEBI used to evaluate market costs or liquidity implications?

How does this proposal align with comparable frameworks of the SEC or ESMA?

What transitional provisions are planned to support small intermediaries or issuers?

How will SEBI measure the effectiveness of this reform post-implementation?

🧩 Additional Instructions

Handle any number of proposals without summary loss. Each proposal must have its own complete analysis block.

Maintain formal, neutral, analytical tone suitable for publication by a think tank.

Format for clean export to Word or Markdown (use clear headings, bullet points, and bold subheadings).

Ensure consistency and readability even for long or complex consultation papers.

If the text contains annexures, FAQs, or appendices, summarize their key regulatory relevance at the end.
{circular_text}"""
                                

                                try:
                                    response = model.generate_content(prompt)
                                    summary = response.text
                                    st.success(" Summary Generated!")
                                    st.subheader("Summary")
                                    st.write(summary)

                                except Exception as e:
                                    st.error(f"An error occurred while generating the summary: {e}")
                    else:
                        st.error("The 'Extracted_Text' column is missing from the data.")
