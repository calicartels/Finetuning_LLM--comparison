"""
Streamlit app for comparing Llama 3 models
"""
import os
import sys
import json
import time
from typing import Dict, Any

import streamlit as st
import plotly.graph_objects as go

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from auths import setup_google_auth
from model_service import load_models, generate_text, compare_models
from data_utils import load_sample_puzzles

# Page configuration
st.set_page_config(
    page_title="Llama 3 Logic Puzzle Comparison",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "generated_responses" not in st.session_state:
    st.session_state.generated_responses = {}

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False


def load_models_with_status():
    """Load models and update session state"""
    with st.spinner("Loading models..."):
        try:
            # Set up Google Cloud authentication
            setup_google_auth()
            
            # Load models
            available_models = load_models()
            
            st.session_state.models_loaded = True
            st.session_state.available_models = list(available_models.keys())
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False


def display_sidebar():
    """Display sidebar with controls and information"""
    st.sidebar.title("🧩 Llama 3 Comparison")
    
    # Model loading
    if not st.session_state.models_loaded:
        if st.sidebar.button("Load Models", use_container_width=True):
            load_models_with_status()
    else:
        st.sidebar.success(f"Models loaded: {', '.join(st.session_state.available_models)}")
    
    st.sidebar.divider()
    
    # Google Cloud Configuration
    st.sidebar.subheader("Google Cloud Configuration")
    
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        st.sidebar.success(f"Using credentials: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
    elif config.DEFAULT_CREDENTIALS_PATH:
        st.sidebar.success(f"Using credentials: {config.DEFAULT_CREDENTIALS_PATH}")
    else:
        st.sidebar.warning("No explicit credentials found. Using application default.")
    
    # Show project ID and region
    st.sidebar.text(f"Project ID: {config.PROJECT_ID}")
    st.sidebar.text(f"Region: {config.REGION}")
    
    st.sidebar.divider()
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    
    params = {
        "max_tokens": st.sidebar.slider("Max Tokens", 64, 2048, 512, 64),
        "temperature": st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05),
        "top_p": st.sidebar.slider("Top P", 0.0, 1.0, 0.8, 0.05),
        "top_k": st.sidebar.slider("Top K", 1, 100, 40, 1),
    }
    
    st.sidebar.divider()
    
    # Project information
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This app compares the original Llama 3 8B model with a "
        "fine-tuned version trained on logic puzzles. "
        "It demonstrates how fine-tuning can improve performance "
        "on specific tasks."
    )
    
    return params


def main():
    """Main application function"""
    st.title("🧩 Llama 3 Logic Puzzle Comparison")
    
    # Display sidebar and get parameters
    params = display_sidebar()
    
    # Main content
    st.markdown(
        """
        Compare the original Llama 3 8B model with a version fine-tuned on logic puzzles.
        Enter a logic puzzle or select one of the samples below.
        """
    )
    
    # Input area
    user_input = st.text_area(
        "Enter your logic puzzle or any other prompt:",
        height=150,
        placeholder="Enter a logic puzzle here..."
    )
    
    # Sample puzzles
    st.subheader("Sample Logic Puzzles")
    
    sample_cols = st.columns(3)
    
    try:
        samples = load_sample_puzzles()
        
        for i, (name, puzzle) in enumerate(samples.items()):
            col_idx = i % 3
            with sample_cols[col_idx]:
                if st.button(name, key=f"sample_{i}", use_container_width=True):
                    st.session_state.user_input = puzzle["question"]
                    st.experimental_rerun()
    except Exception as e:
        st.warning(f"Could not load sample puzzles: {str(e)}")
    
    # Generate button
    generate_col1, generate_col2 = st.columns([1, 3])
    
    with generate_col1:
        if not st.session_state.models_loaded:
            st.warning("Please load models first")
            generate_button = st.button(
                "Load Models & Compare", 
                type="primary", 
                disabled=not user_input,
                use_container_width=True
            )
        else:
            generate_button = st.button(
                "Compare Models", 
                type="primary", 
                disabled=not user_input,
                use_container_width=True
            )
    
    # Generation
    if generate_button and user_input:
        if not st.session_state.models_loaded:
            load_success = load_models_with_status()
            if not load_success:
                st.stop()
        
        with st.spinner("Generating responses..."):
            comparison = compare_models(
                user_input,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"]
            )
            
            st.session_state.generated_responses = comparison
    
    # Display results
    if "generated_responses" in st.session_state and st.session_state.generated_responses:
        responses = st.session_state.generated_responses
        
        st.subheader("Model Comparison")
        
        if "warning" in responses:
            st.warning(responses["warning"])
        
        if "error" in responses:
            st.error(responses["error"])
            st.stop()
        
        # Display responses in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Base Llama 3 8B")
            if "base_model" in responses and "response_only" in responses["base_model"]:
                base_response = responses["base_model"]["response_only"]
                st.markdown(f"```\n{base_response}\n```")
                
                generation_time = responses["base_model"].get("generation_time", 0)
                if generation_time:
                    st.caption(f"Generated in {generation_time:.2f} seconds")
        
        with col2:
            st.markdown("### Fine-tuned Llama 3 8B")
            if "finetuned_model" in responses and "response_only" in responses["finetuned_model"]:
                tuned_response = responses["finetuned_model"]["response_only"]
                st.markdown(f"```\n{tuned_response}\n```")
                
                generation_time = responses["finetuned_model"].get("generation_time", 0)
                if generation_time:
                    st.caption(f"Generated in {generation_time:.2f} seconds")
        
        # Metrics or analysis section
        st.subheader("Analysis")
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            base_length = len(responses.get("base_model", {}).get("response_only", ""))
            st.metric("Base Model Response Length", base_length)
        
        with metric_cols[1]:
            tuned_length = len(responses.get("finetuned_model", {}).get("response_only", ""))
            st.metric("Fine-tuned Model Response Length", tuned_length)
        
        with metric_cols[2]:
            length_diff = tuned_length - base_length
            st.metric("Length Difference", length_diff, f"{length_diff:+}")
        
        # Optionally add a visualization
        if "base_model" in responses and "finetuned_model" in responses:
            if "generation_time" in responses["base_model"] and "generation_time" in responses["finetuned_model"]:
                base_time = responses["base_model"]["generation_time"]
                tuned_time = responses["finetuned_model"]["generation_time"]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Base Model", "Fine-tuned Model"],
                    y=[base_time, tuned_time],
                    text=[f"{base_time:.2f}s", f"{tuned_time:.2f}s"],
                    textposition="auto",
                    marker_color=["#1f77b4", "#ff7f0e"]
                ))
                fig.update_layout(
                    title="Generation Time Comparison",
                    xaxis_title="Model",
                    yaxis_title="Time (seconds)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    # Auto-load models if run directly
    if not st.session_state.models_loaded:
        load_models_with_status()
    
    main()