import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from contextlib import redirect_stdout
from openai import OpenAI
from dotenv import load_dotenv

# --- OpenAI and Langchain Imports ---
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# =================================================================================
# ALL HELPER FUNCTIONS
# =================================================================================

# --- FROM missing_vals.ipynb ---

def analyze_missing_values(df, drop_threshold=0.5):
    """Analyzes missing values in a DataFrame and suggests imputation strategies."""
    suggestions = {}
    for col in df.columns:
        missing_pct = df[col].isna().mean()
        dtype = df[col].dtype
        suggestion = None
        if missing_pct == 0:
            continue
        if missing_pct > drop_threshold:
            suggestion = "Drop column (too many missing values)"
        else:
            if pd.api.types.is_numeric_dtype(dtype):
                n_unique = df[col].nunique(dropna=True)
                if n_unique < 15:
                    suggestion = "Mode imputation (numeric categorical)"
                else:
                    non_null = df[col].dropna()
                    if len(non_null) < 10:
                        suggestion = "Median imputation (small sample)"
                    else:
                        skewness = non_null.skew()
                        suggestion = "Mean imputation" if abs(skewness) < 1 else "Median imputation (skewed)"
            elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                n_unique = df[col].nunique(dropna=True)
                if n_unique <= 10:
                    suggestion = "Mode imputation (most frequent)"
                else:
                    suggestion = "Impute with 'Unknown' or predictive model"
            elif pd.api.types.is_bool_dtype(dtype):
                suggestion = "Mode imputation (True/False)"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                suggestion = "Forward/Backward fill or interpolation (time series)"
            else:
                suggestion = "Custom handling needed"
        suggestions[col] = {
            "dtype": str(dtype),
            "missing_pct": round(missing_pct, 3),
            "suggestion": suggestion
        }
    return pd.DataFrame.from_dict(suggestions, orient="index")

def missing_vals(df, user_query, api_key):
    """Generates Python code to handle missing values based on user query."""
    suggestions = analyze_missing_values(df)
    helper_docs = """ Helper functions available:

    auto_impute(df, drop_threshold=0.5): Automatically imputes missing values in a DataFrame based on simple best-practice heuristics.

    impute_col(series, strategy): Impute missing values in a pandas Series using the specified strategy. and returns modified series

    Supported Strategies: 'mean', 'median', 'mode', 'unknown', 'ffill', 'bfill'
    Examples:

    df = auto_impute(df)

    df[col] = impute_col(df[col], 'mean')
    """
    messages = [
        SystemMessage(content=helper_docs),
        SystemMessage(content=f"""
        You are a data cleaning agent.
        Dataset info: Shape: {df.shape}, Sample: {df.head(3).to_string()}
        Imputation suggestions: {suggestions if not suggestions.empty else "No Missing Values!"}
        Libraries available: pd (pandas), np (numpy)
        Rules:

        Return only executable Python code.

        Do not add explanations or markdown ```python blocks.

        The code you write will be executed using exec().

        The dataframe is already defined as df.

        Use print() to communicate with the user (e.g., print("Missing values imputed.")).

        Your code must modify the dataframe df in place or return it.
        """),
        HumanMessage(content=f"User request: {user_query}")
    ]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=api_key)
    response = llm.invoke(messages)
    return response.content.strip()

# --- FROM summaries.ipynb ---

def numerical_summary(df, columns=None, metrics=None):
    """Generates and prints a numerical summary of the DataFrame."""
    if not isinstance(df, pd.DataFrame): return pd.DataFrame()
    if columns:
        columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    else:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not columns: return pd.DataFrame()
    filtered_df = df[columns]
    default_metrics = {
        "count": filtered_df.count(), "missing": filtered_df.isna().sum(),
        "mean": filtered_df.mean(), "std": filtered_df.std(), "var": filtered_df.var(),
        "min": filtered_df.min(), "q1": filtered_df.quantile(0.25),
        "median": filtered_df.median(), "q3": filtered_df.quantile(0.75),
        "max": filtered_df.max(), "skew": filtered_df.skew(),
        "iqr": filtered_df.quantile(0.75) - filtered_df.quantile(0.25),
        "range": filtered_df.max() - filtered_df.min()
    }
    if metrics:
        metrics = {m.lower() for m in metrics}
        output_cols = {"count": default_metrics["count"]}
        for d in default_metrics:
            if d != "count" and d in metrics:
                output_cols[d] = default_metrics[d]
    else:
        output_cols = default_metrics
    output_df = pd.DataFrame(output_cols)
    print(output_df.to_markdown()) # Use markdown for better table formatting
    return df

def categorical_summary(df, columns=None, metrics=None):
    """Generates and prints a categorical summary of the DataFrame."""
    if not isinstance(df, pd.DataFrame): return pd.DataFrame()
    if columns:
        columns = [col for col in columns if col in df.columns]
    else:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not columns: return pd.DataFrame()
    filtered_df = df[columns]
    default_metrics = {
        "count": filtered_df.count(), "missing": filtered_df.isna().sum(),
        "nunique": filtered_df.nunique(dropna=True),
        "mode": filtered_df.apply(lambda col: col.mode(dropna=True).iloc if not col.mode(dropna=True).empty else np.nan),
        "top_freq": filtered_df.apply(lambda col: col.value_counts(dropna=True).iloc if not col.value_counts(dropna=True).empty else 0)
    }
    if metrics:
        metrics = {m.lower() for m in metrics}
        output_cols = {"count": default_metrics["count"]}
        for d in default_metrics:
            if d != "count" and d in metrics:
                output_cols[d] = default_metrics[d]
    else:
        output_cols = default_metrics
    output_df = pd.DataFrame(output_cols)
    print(output_df.to_markdown()) # Use markdown for better table formatting
    return df

def get_summaries(df, user_query, api_key):
    """Generates Python code for data summaries based on user query."""
    helper_docs = """Helper functions available:

    numerical_summary(df, columns=None, metrics=None): Prints a numeric summary for given columns.

    metrics: subset of ["count","missing","mean","std","min","median","q1","q3","max","skew","range", "iqr"]

    categorical_summary(df, columns=None, metrics=None): Prints a categorical summary for given columns.

    metrics: subset of ['count', 'missing', 'nunique', 'mode', 'top_freq']
    Examples:

    "Find cardinality of categorical columns" -> categorical_summary(df, metrics=['nunique'])

    "summary of all data" -> numerical_summary(df); categorical_summary(df)
    """
    messages = [
        SystemMessage(content=helper_docs),
        SystemMessage(content=f"""
        You are a data summarization agent.
        Dataset info: Shape: {df.shape}, Sample: {df.head(3).to_string()}
        Libraries available: pd (pandas), np (numpy)
        Rules:

        Return only executable Python code.

        Do not add explanations or markdown ```python blocks.

        The code you write will be executed using exec().

        The dataframe is already defined as df.

        Use helper functions when appropriate.

        Use print() to generate the summary or response for the user.
        """),
        HumanMessage(content=f"User request: {user_query}")
    ]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=api_key)
    response = llm.invoke(messages)
    return response.content.strip()

# --- Data Cleaning Recommendation and Code Generation Functions ---

def recommend_cleaning_steps(df: pd.DataFrame, api_key: str) -> str:
    """
    Analyzes the DataFrame and returns a list of cleaning recommendations.
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=api_key)
    
    # For simplicity, this example will analyze a string representation of the DataFrame's info and head.
    # The original chunking logic is good for very large files but can be simplified for this example.
    df_info_buffer = io.StringIO()
    df.info(buf=df_info_buffer)
    df_info = df_info_buffer.getvalue()
    
    analysis_prompt = f"""
    You are a data quality analyst. Given the following summary and head of a dataset,
    identify potential data quality issues and suggest cleaning steps. Focus on:
    - Inconsistent data types in columns (e.g., numbers stored as objects).
    - Obvious outliers or anomalies in the sample data.
    - Mixed formatting (e.g., dates, strings).
    - Redundant or unnecessary columns/rows.
    - Structural issues.
    - Missing values that need handling.
    
    List the issues and your recommended cleaning steps as a clear, actionable list.

    DataFrame Info:
    {df_info}

    DataFrame Head:
    {df.head().to_string()}
    """
    
    messages = [HumanMessage(content=analysis_prompt)]
    response = llm.invoke(messages)
    return response.content.strip()

def generate_cleaning_code(df: pd.DataFrame, recommendations: str, api_key: str, error_message: str = None) -> str:
    """
    Generates a Python function to clean the DataFrame based on recommendations,
    optionally correcting it based on a previous error.
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=api_key)
    
    # Add error context to the prompt if an error message is provided
    error_context = ""
    if error_message:
        error_context = f"""
        IMPORTANT: The previously generated code failed with the following error.
        You MUST fix the code to resolve this issue. Do not repeat the same mistake.
        Error Message: {error_message}
        """

    prompt = f"""
    You are a Python data science expert. Based on the provided DataFrame summary, cleaning recommendations,
    and a potential error from a previous attempt, write a single Python function called `clean_data`.
    This function must accept a DataFrame `df` as input and return the cleaned DataFrame.

    {error_context}

    Rules:
    - Only output the Python code for the function.
    - Do not include any explanations, comments, or markdown formatting like ```python.
    - The function must be named `clean_data`.
    - The function must take `df` as an argument and `return` the modified `df`.
    - Ensure the code is robust and handles potential errors (e.g., use .get() for dictionaries, check if columns exist before dropping).

    DataFrame Info:
    {df.info()}

    DataFrame Head:
    {df.head().to_string()}

    Cleaning Recommendations:
    {recommendations}
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content.strip()

# =================================================================================
# STREAMLIT APPLICATION
# =================================================================================

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your Data", layout="wide")
    st.title("Chat with your Data 📊")

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.getenv("OPENAI_KEY")

    # --- Sidebar for Uploading and API Key ---
    with st.sidebar:
        st.header("Setup")
        uploaded_files = st.file_uploader(
            "Choose CSV files", type="csv", accept_multiple_files=True
        )

        if uploaded_files:
            try:
                dataframes = [pd.read_csv(file) for file in uploaded_files]
                st.session_state.df = pd.concat(dataframes, ignore_index=True)
                st.success("CSV files uploaded successfully!")
                with st.expander("Data Preview"):
                    st.dataframe(st.session_state.df.head())
            except Exception as e:
                st.error(f"Error loading files: {e}")
                st.session_state.df = None

    # --- Main Chat Interface ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question or type 'clean my data'..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if st.session_state.df is None:
            st.warning("Please upload a CSV file to begin.")
            st.stop()
            
        # --- Agent Logic: Routing and Execution ---
        with st.chat_message("assistant"):
            response_content = ""
            try:
                prompt_lower = prompt.lower()

                # --- ROBUST: End-to-End Cleaning Flow with Retry Logic ---
                if "clean" in prompt_lower:
                    with st.spinner("Analyzing data for cleaning recommendations..."):
                        recommendations = recommend_cleaning_steps(st.session_state.df, st.session_state.api_key)
                        st.markdown(f"**Data Cleaning Recommendations:**\n\n{recommendations}")
                        st.session_state.messages.append({"role": "assistant", "content": f"**Data Cleaning Recommendations:**\n\n{recommendations}"})
                    
                    max_retries = 3
                    last_error = None
                    cleaned_df = None

                    for attempt in range(max_retries):
                        st.info(f"Attempt {attempt + 1} of {max_retries} to clean the data...")
                        try:
                            with st.spinner(f"Generating cleaning code (Attempt {attempt + 1})..."):
                                # Pass the last error message to the code generator on subsequent attempts
                                cleaning_code = generate_cleaning_code(
                                    st.session_state.df, 
                                    recommendations, 
                                    st.session_state.api_key, 
                                    error_message=last_error
                                )
                            
                            with st.spinner(f"Executing cleaning code (Attempt {attempt + 1})..."):
                                local_scope = {}
                                # Execute the generated function definition
                                exec(cleaning_code, globals(), local_scope)
                                clean_data_func = local_scope['clean_data']
                                
                                # Run the generated function
                                cleaned_df = clean_data_func(st.session_state.df.copy())
                                
                                # If successful, update the session state and break the loop
                                st.session_state.df = cleaned_df
                                response_content = "✅ Data has been cleaned successfully!"
                                st.success(response_content)
                                st.session_state.messages.append({"role": "assistant", "content": response_content})

                                # Provide download button
                                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Cleaned CSV",
                                    data=csv,
                                    file_name="cleaned_data.csv",
                                    mime="text/csv",
                                )
                                break  # Exit the loop on success

                        except Exception as e:
                            last_error = str(e)
                            st.warning(f"Attempt {attempt + 1} failed: {last_error}")
                            if attempt == max_retries - 1:
                                # Final attempt failed
                                response_content = f"❌ Failed to clean the data after {max_retries} attempts. Last error: {last_error}"
                                st.error(response_content)
                                st.session_state.messages.append({"role": "assistant", "content": response_content})
                    st.stop()

                # --- Existing Routing Logic ---
                elif any(word in prompt_lower for word in ['missing', 'impute', 'null', 'nan', 'drop']):
                    tool_function = missing_vals
                elif any(word in prompt_lower for word in ['summary', 'describe', 'stats', 'mean', 'median', 'mode']):
                    tool_function = get_summaries
                else:
                    # Fallback to a general Q&A if no tool matches
                    st.info("No specific tool matched. Forwarding to general chat.")
                    response_content = "Please ask about cleaning, missing values, or summaries."

                if 'tool_function' in locals():
                    generated_code = tool_function(st.session_state.df, prompt, st.session_state.api_key)
                    
                    output_buffer = io.StringIO()
                    with redirect_stdout(output_buffer):
                        temp_df = st.session_state.df.copy()
                        local_scope = {'df': temp_df, 'pd': pd, 'np': np}
                        exec(generated_code, globals(), local_scope)
                        st.session_state.df = local_scope.get('df', st.session_state.df)

                    response_content = output_buffer.getvalue()
                    if not response_content:
                        response_content = "Action completed. The DataFrame may have been updated."

            except Exception as e:
                response_content = f"An unexpected error occurred in the main application logic: {e}"

            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()