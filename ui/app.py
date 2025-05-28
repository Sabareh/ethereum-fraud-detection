import streamlit as st
import warnings
import sys
import os
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import page modules
from pages.home_page import render_home_page
from pages.eda_page import render_eda_page
from pages.unsupervised_page import render_unsupervised_page
from pages.supervised_page import render_supervised_page
from pages.explainability_page import render_explainability_page
from pages.model_analysis_page import render_model_analysis_page
from pages.realtime_prediction_page import render_realtime_prediction_page
from pages.data_management_page import render_data_management_page
from model_utils import load_data

BASE_DIR = Path(__file__).resolve().parent.parent

# Configure page settings - disable file watcher and remove default navigation
st.set_page_config(
    page_title="Ethereum Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit's default file navigation and pages
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    header {visibility: hidden;}
    
    /* Hide the default pages navigation */
    section[data-testid="stSidebar"] > div:first-child > div:first-child > div:first-child {
        display: none;
    }
    
    /* Hide pages navigation if it appears anywhere */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Hide any file navigation elements */
    .css-1d391kg {
        display: none;
    }
    
    /* Hide the pages selector if visible */
    .css-1rs6os {
        display: none;
    }
    
    /* Additional rules to ensure navigation is completely hidden */
    .css-17eq0hr {
        display: none;
    }
    
    .css-1544g2n {
        display: none;
    }
    
    /* Hide any remaining navigation elements */
    [data-testid="stSidebarNavItems"] {
        display: none !important;
    }
    
    [data-testid="stSidebarNavLink"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add error catching for the entire app
try:
    def main():
        # Suppress warnings in streamlit
        warnings.filterwarnings("ignore")
        
        # Initialize session state for page navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home"
        
        # Sidebar navigation
        with st.sidebar:
            st.title("🔍 Ethereum Fraud Detection")
            st.markdown("---")
            
            # Navigation buttons instead of selectbox to avoid file navigation
            st.subheader("Navigation")
            
            # Create navigation buttons
            nav_buttons = {
                "🏠 Home": "Home",
                "📊 Data Management": "Data Management", 
                "📈 EDA Overview": "EDA Overview",
                "🎯 Unsupervised Learning": "Unsupervised",
                "🤖 Supervised Learning": "Supervised",
                "🔬 Model Analysis": "Model Analysis",
                "⚡ Real-time Prediction": "Real-time Prediction",
                "🧠 Explainability": "Explainability"
            }
            
            # Create navigation using buttons
            for display_name, page_name in nav_buttons.items():
                if st.button(display_name, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    st.experimental_rerun()
            
            # Highlight current page
            current_display = None
            for display_name, page_name in nav_buttons.items():
                if page_name == st.session_state.current_page:
                    current_display = display_name
                    break
            
            if current_display:
                st.markdown(f"**Currently viewing:** {current_display}")
            
            st.markdown("---")
            
            # Add quick info about current page
            page_info = {
                "Home": "Welcome to the Ethereum Fraud Detection System",
                "Data Management": "Upload and manage your datasets",
                "EDA Overview": "Exploratory Data Analysis and visualization",
                "Unsupervised": "Clustering and anomaly detection models",
                "Supervised": "Classification models for fraud detection",
                "Model Analysis": "Advanced model analysis and comparison",
                "Real-time Prediction": "Live fraud detection predictions",
                "Explainability": "Model interpretability and feature analysis"
            }
            
            st.info(f"**About:** {page_info.get(st.session_state.current_page, 'Unknown page')}")
            
            # Add system status
            st.markdown("---")
            st.subheader("System Status")
            try:
                # Quick data check
                X, y = load_data(use_synthetic=True)
                st.success(f"✅ Data loaded: {X.shape[0]} samples")
                st.info(f"📊 Features: {X.shape[1]}")
                st.info(f"🎯 Fraud rate: {y.mean():.2%}")
            except Exception as e:
                st.error(f"❌ Data issue: {str(e)[:50]}...")

        # Add a data loading section with error handling
        @st.cache_data
        def get_data(use_synthetic=True):
            """Load data with caching and error handling."""
            try:
                with st.spinner("Loading data..."):
                    X, y = load_data(use_synthetic=use_synthetic)
                    return X, y, None
            except Exception as e:
                return None, None, str(e)

        # Main content area
        current_page = st.session_state.current_page
        
        # Load data for pages that need it
        data_dependent_pages = ["Unsupervised", "Supervised", "Model Analysis", "Real-time Prediction", "Explainability"]
        
        if current_page in data_dependent_pages:
            X, y, data_error = get_data()
            if data_error:
                st.error(f"⚠️ Error loading data: {data_error}")
                st.warning("🔄 Using synthetic data instead. Some functionality may be limited.")
                X, y, _ = get_data(use_synthetic=True)
        else:
            X, y = None, None

        # Route to appropriate page with error handling
        try:
            if current_page == "Home":
                render_home_page()
            elif current_page == "Data Management":
                render_data_management_page()
            elif current_page == "EDA Overview":
                render_eda_page()
            elif current_page == "Unsupervised":
                if X is not None and y is not None:
                    render_unsupervised_page(X, y)
                else:
                    st.error("❌ Cannot load unsupervised page: Data not available")
            elif current_page == "Supervised":
                if X is not None and y is not None:
                    render_supervised_page(X, y)
                else:
                    st.error("❌ Cannot load supervised page: Data not available")
            elif current_page == "Model Analysis":
                render_model_analysis_page()
            elif current_page == "Real-time Prediction":
                render_realtime_prediction_page()
            elif current_page == "Explainability":
                render_explainability_page()
            else:
                st.error(f"❌ Page '{current_page}' not found!")
                st.info("Please select a valid page from the sidebar.")
                
        except Exception as page_error:
            st.error(f"❌ Error loading page '{current_page}': {str(page_error)}")
            st.warning("Please try refreshing the page or selecting a different page.")
            
            # Show error details in expander for debugging
            with st.expander("🔧 Error Details (for debugging)"):
                import traceback
                st.code(traceback.format_exc())
                
                # Quick page navigation fallback
                st.subheader("Quick Navigation")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("🏠 Home", key="emergency_home"):
                        st.session_state.current_page = "Home"
                        st.experimental_rerun()
                with col2:
                    if st.button("📊 Data", key="emergency_data"):
                        st.session_state.current_page = "Data Management"
                        st.experimental_rerun()
                with col3:
                    if st.button("📈 EDA", key="emergency_eda"):
                        st.session_state.current_page = "EDA Overview"
                        st.experimental_rerun()
                with col4:
                    if st.button("🤖 Models", key="emergency_models"):
                        st.session_state.current_page = "Supervised"
                        st.experimental_rerun()

        # Custom footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 0.8em; padding: 20px;'>
                🔍 Ethereum Fraud Detection System | Built with Streamlit | 
                <a href='https://github.com/Sabareh/ethereum-fraud-detection' target='_blank' style='color: #666;'>GitHub Repository</a>
            </div>
            """, 
            unsafe_allow_html=True
        )

    if __name__ == "__main__":
        # Run the app with comprehensive error handling
        try:
            main()
        except Exception as e:
            st.error(f"🚨 Critical application error: {str(e)}")
            
            # Emergency navigation
            st.subheader("🆘 Emergency Navigation")
            st.warning("The application encountered a critical error. Try these recovery options:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Reload Application", key="emergency_reload"):
                    st.experimental_rerun()
            with col2:
                if st.button("🏠 Go to Home", key="emergency_goto_home"):
                    st.session_state.current_page = "Home"
                    st.experimental_rerun()
            with col3:
                if st.button("🗑️ Clear Cache", key="emergency_clear_cache"):
                    st.cache_data.clear()
                    st.experimental_rerun()
            
            # Show full error for debugging
            with st.expander("🔧 Full Error Details"):
                import traceback
                st.code(traceback.format_exc())

except ImportError as import_error:
    st.error(f"🚨 Import Error: {str(import_error)}")
    st.warning("Some required modules could not be imported. Please check your installation.")
    
    # Show missing modules
    st.subheader("Troubleshooting Steps:")
    st.markdown("""
    1. **Check Python Environment**: Ensure you're using the correct Python environment
    2. **Install Requirements**: Run `pip install -r requirements.txt`
    3. **Check File Structure**: Ensure all page modules exist in the correct directories
    4. **Restart Application**: Try restarting the Streamlit application
    """)
    
except Exception as critical_error:
    st.error(f"🚨 Critical startup error: {str(critical_error)}")
    
    # Basic recovery interface
    st.subheader("🆘 System Recovery")
    st.markdown("The application failed to start properly. This might be due to:")
    st.markdown("""
    - Missing dependencies
    - File permission issues  
    - Configuration problems
    - Data loading issues
    """)
    
    if st.button("🔄 Attempt Recovery", key="critical_recovery"):
        st.experimental_rerun()
