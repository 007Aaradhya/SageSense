import streamlit as st
from pages import Home, Dataset_Load, Train_Models, Upload_Predict, Visualization

# Custom Page Config
st.set_page_config(
    page_title="SageSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme (Neon Cyberpunk Style)
st.markdown(f"""
    <style>
    :root {{
        --primary: #6e00ff;
        --secondary: #ff00c1;
        --dark: #0a0a1a;
        --light: #e0f8ff;
        --accent: #00f7ff;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, var(--dark) 0%, #1a0033 100%);
        color: var(--light);
        font-family: 'Segoe UI', sans-serif;
    }}
    
    .sidebar .sidebar-content {{
        background: rgba(10, 10, 26, 0.9) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid var(--primary);
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: var(--accent) !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(0, 247, 255, 0.3);
    }}
    
    .stButton>button {{
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        border: none;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        padding: 0.8rem 2rem;
        box-shadow: 0 0 15px rgba(110, 0, 255, 0.5);
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(255, 0, 193, 0.7);
    }}
    
    .stRadio>div {{
        background-color: rgba(30, 30, 60, 0.7) !important;
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid var(--primary);
    }}
    
    .stRadio>div>label>div:first-child {{
        color: var(--accent) !important;
    }}
    
    .stTextInput>div>div>input {{
        background-color: rgba(20, 20, 40, 0.7) !important;
        color: var(--light) !important;
        border: 1px solid var(--primary) !important;
    }}
    
    .stDataFrame {{
        background-color: rgba(20, 20, 40, 0.7) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 10px;
    }}
    
    /* Glowing divider */
    .divider {{
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), var(--secondary), transparent);
        margin: 1.5rem 0;
        box-shadow: 0 0 10px var(--primary);
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: var(--dark);
    }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(var(--primary), var(--secondary));
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# Session State Management
if "page" not in st.session_state:
    st.session_state.page = "Home"
    st.session_state.theme = "dark"

# Cyberpunk Sidebar
with st.sidebar:
    # Animated Header
    st.markdown("""
    <h1 style="text-align: center; 
               background: linear-gradient(90deg, #6e00ff, #ff00c1);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               animation: gradient 5s ease infinite;
               background-size: 200% 200%;
               margin-bottom: 0;">
        SAGE<span style="color: #00f7ff">SENSE</span>
    </h1>
    <p style="text-align: center; color: #00f7ff; margin-top: 0;">AI MODEL WORKBENCH</p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Navigation with Icons
    nav_options = {
        "Home": "🏠",
        "Dataset Load": "📊", 
        "Train Models": "⚙️",
        "Upload & Predict": "🔮",
        "Visualization": "📈"
    }
    
    page = st.radio(
        "NAVIGATION",
        options=list(nav_options.keys()),
        format_func=lambda x: f"{nav_options[x]} {x}",
        label_visibility="collapsed"
    )
    st.session_state.page = page
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Quick Tips
    st.markdown("""
    <div style="background: rgba(110, 0, 255, 0.1); 
                padding: 1rem; 
                border-radius: 10px;
                border-left: 3px solid var(--accent);">
        <h4 style="color: var(--accent) !important;">PRO TIPS</h4>
        <ul style="color: var(--light);">
            <li>Start with clean datasets</li>
            <li>Try different model architectures</li>
            <li>Visualize before predictions</li>
            <li>Save your trained models</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; font-size: 0.8rem;">
        <p style="color: var(--accent);">v2.0.0</p>
        <p style="color: #6e00ff;">NEURAL ARCHITECTURE</p>
    </div>
    """, unsafe_allow_html=True)

# Main Content Area
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url('https://images.unsplash.com/photo-1639762681057-408e52192e55?q=80&w=2832&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    
    .main .block-container {{
        background: rgba(10, 10, 26, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        border: 1px solid var(--primary);
        box-shadow: 0 0 30px rgba(110, 0, 255, 0.3);
    }}
    </style>
""", unsafe_allow_html=True)

# Page Routing
if st.session_state.page == "Home":
    Home.show()
elif st.session_state.page == "Dataset Load":
    Dataset_Load.show()
elif st.session_state.page == "Train Models":
    Train_Models.show() 
elif st.session_state.page == "Upload & Predict":
    Upload_Predict.show()
elif st.session_state.page == "Visualization":
    Visualization.show()

# Floating Terminal Effect
st.markdown("""
    <div style="position: fixed; 
                bottom: 20px; 
                right: 20px; 
                background: rgba(0, 0, 0, 0.7); 
                color: #00ff00; 
                padding: 0.5rem 1rem;
                border-radius: 5px;
                font-family: monospace;
                border: 1px solid #00ff00;
                font-size: 0.8rem;">
        SYSTEM STATUS: OPERATIONAL
    </div>
""", unsafe_allow_html=True)