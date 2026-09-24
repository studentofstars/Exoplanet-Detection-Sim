# Exoplanet Detection Simulation using Radial Velocity Method
# This code fetches data from the NASA Exoplanet Archive, calculates radial velocity curves,
# visualizes the data in 3D, and allows users to interact with the data using Streamlit.
# Required Libraries
# pip install streamlit plotly pandas numpy requests astropy groq

# Import necessary libraries   
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from astropy.constants import G
from astropy import units as u
from groq import Groq
import time
import os

# Configure Groq API
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    groq_client = None

# Initialize session state for rate limiting
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0  

# Step 1: Fetch Exoplanet Data from NASA Exoplanet Archive
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_exoplanet_data(limit=10000):
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = f"""
    SELECT TOP {limit}
        pl_name,
        hostname,
        pl_bmasse,
        pl_orbper,
        pl_orbsmax,
        pl_orbeccen,
        st_mass,
        st_teff,
        pl_rade
    FROM
        ps
    WHERE
        pl_bmasse IS NOT NULL AND
        pl_orbper IS NOT NULL AND
        pl_orbsmax IS NOT NULL AND
        st_mass IS NOT NULL
    ORDER BY
        pl_orbper ASC
    """
    params = {
        "query": query,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, timeout=10)  # Add timeout
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Convert numeric columns immediately
            numeric_columns = ['pl_bmasse', 'pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'st_mass', 'st_teff', 'pl_rade']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop NA rows immediately
            df = df.dropna(subset=['pl_bmasse', 'pl_orbper', 'pl_orbsmax', 'st_mass'])
            return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# Step 2: Calculate Radial Velocity Amplitude (K)
def calculate_radial_velocity(planet_mass, star_mass, orbital_period, eccentricity=0):
    planet_mass = planet_mass * u.M_earth  # Planet mass in Earth masses
    star_mass = star_mass * u.M_sun        # Star mass in Solar masses
    orbital_period = orbital_period * u.day # Orbital period in days

    orbital_period = orbital_period.to(u.second)  # Convert orbital period to seconds
    G_const = G.value  # Gravitational constant (m^3 kg^-1 s^-2)

    planet_mass_kg = planet_mass.to(u.kg).value  # Convert planet mass to kg
    star_mass_kg = star_mass.to(u.kg).value      # Convert star mass to kg

    # Calculate radial velocity amplitude (K) in m/s
    K = ((2 * np.pi * G_const) / orbital_period.value)**(1/3) * (planet_mass_kg) / (star_mass_kg**(2/3)) / np.sqrt(1 - eccentricity**2)
    return K  # in m/s


# Step 3: Generate Radial Velocity Curve
def generate_radial_velocity_curve(K, P, time_span):
    time_points = np.linspace(0, time_span, 1000)  # Time points (days)
    velocity = K * np.sin(2 * np.pi * time_points / P)  # Radial velocity at each time point
    return time_points, velocity
#Step 4: Calculate habitable zone
def calculate_habitable_zone(star_teff): # Constants for habitable zone calculation (Kopparapu et al. 2014) 
    S_eff_sun = np.array([1.776, 0.320]) 
    a = np.array([0.013, 0.094]) 
    b = np.array([2.04e-4, 1.73e-4]) 
    c = np.array([-2.89e-8, -5.44e-9]) 
    T_star = star_teff # Effective temperature of the star 
    T_sun = 5778 # Effective temperature of the sun 
    L = (star_teff / T_sun)**4 # Luminosity of the star in terms of solar luminosity
    # Calculate the inner and outer boundaries of the habitable zone 
    denominator_inner = S_eff_sun[0] + a[0] * (T_star - T_sun) + b[0] * (T_star - T_sun)**2 + c[0] * (T_star - T_sun)**3
    denominator_outer = S_eff_sun[1] + a[1] * (T_star - T_sun) + b[1] * (T_star - T_sun)**2 + c[1] * (T_star - T_sun)**3
    # Ensure denominators are positive to avoid sqrt of negative numbers
    denominator_inner = np.maximum(denominator_inner, 1e-10)
    denominator_outer = np.maximum(denominator_outer, 1e-10)
    r_inner = np.sqrt(L / denominator_inner)
    r_outer = np.sqrt(L / denominator_outer)
    return r_inner, r_outer
    


# Step 5: Streamlit App Setup
st.title("Exoplanet Detection Simulation")

# Add tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Radial Velocity Curves", "Planet Details", "3D Orbits", "Habitable Zone Calculation", "Real-Time Data", "AI Assistant"])
#Sidebar for Tab 1
with st.sidebar:
    st.header("Adjust Filter Parameters")
    st.markdown(" Set your parameters to filter out the curves ")
    dataset_count = st.number_input('Enter the number of datasets to import:', min_value=1, max_value=10000, value=10)

    df = fetch_exoplanet_data(limit=dataset_count)
    if df is not None:
        df = df.dropna(subset=['pl_bmasse', 'pl_orbper', 'pl_orbsmax', 'pl_orbeccen', 'st_mass'])
    # Filter options
    min_mass = st.slider('Select minimum planet mass (Earth Masses)', min_value=float(df['pl_bmasse'].min()), max_value=float(df['pl_bmasse'].max()), value=float(df['pl_bmasse'].min()))
    max_mass = st.slider('Select maximum planet mass (Earth Masses)', min_value=min_mass, max_value=float(df['pl_bmasse'].max()), value=float(df['pl_bmasse'].max()))
    min_period = st.slider('Select minimum orbital period (days)', min_value=float(df['pl_orbper'].min()), max_value=float(df['pl_orbper'].max()), value=float(df['pl_orbper'].min()))
    max_period = st.slider('Select maximum orbital period (days)', min_value=min_period, max_value=float(df['pl_orbper'].max()), value=float(df['pl_orbper'].max()))
    # Slider for eccentricity
    eccentricity = st.slider('Eccentricity', min_value=0.0, max_value=1.0, step=0.01, value=0.0)
  

    
with tab1:
    st.header("Radial Velocity Method")

    # Brief explanation of the Radial Velocity Method
    st.markdown("""
    The **Radial Velocity Method** is used to detect exoplanets by observing the wobble of stars due to gravitational pulls from orbiting planets. 
    This method helps determine important parameters such as the mass of the planet, its orbital period, and the distance from the star.
    The data for this graph is fetched directly from the NASA Exoplanet Archive in real time and is processed to generate the plots as shown.
    The radial velocity curves represent the variations in the star's radial velocity caused by the gravitational influence of an orbiting exoplanet.
    """)
   
    if df is not None:
        filtered_df = df[(df['pl_bmasse'] >= min_mass) & (df['pl_bmasse'] <= max_mass) & (df['pl_orbper'] >= min_period) & (df['pl_orbper'] <= max_period)]

        if filtered_df.empty:
            st.write("No planets match your filters!")
        else:
            fig = go.Figure()

            for index, planet in filtered_df.iterrows():
                planet_name = planet['pl_name']
                star_name = planet['hostname']
                planet_mass = planet['pl_bmasse']
                orbital_period = planet['pl_orbper']
                star_mass = planet['st_mass']

                # Calculate radial velocity amplitude
                K = calculate_radial_velocity(planet_mass, star_mass, orbital_period, eccentricity)

                # Generate radial velocity curve
                time_span = orbital_period * 2
                time_points, velocity = generate_radial_velocity_curve(K, orbital_period, time_span)

                # Add the curve to the Plotly figure
                fig.add_trace(go.Scatter(x=time_points, y=velocity, mode='lines', name=f'{planet_name} ({star_name})'))

            fig.update_layout(title='Radial Velocity Curves', xaxis_title='Time (days)', yaxis_title='Radial Velocity (m/s)')
            st.plotly_chart(fig)

    st.subheader("Key Formulas Used")

    # Formula for Radial Velocity Amplitude (K)
    st.markdown("""
    The radial velocity amplitude (K) can be calculated using the formula:
    
    $$K = \\frac{2 \\pi G}{P} \\cdot \\frac{M_p}{M_*^{2/3}} \\cdot \\frac{1}{\\sqrt{1 - e^2}}$$
    
    where:
    - \(K\) = Radial velocity amplitude (m/s)
    - \(G\) = Gravitational constant ($$(6.67430 \\cdot 10^{-11} \, m^3 \, kg^{-1} \, s^{-2})$$)
    - \(P\) = Orbital period (in seconds)
    - \(M_p\) = Mass of the planet (in kg)
    - \(M_*\) = Mass of the star (in kg)
    - \(e\) = Eccentricity of the orbit (dimensionless)
    """)

    # Explanation of each parameter
    st.write("### Significance of Parameters:")
    st.write("""
    - **Radial Velocity Amplitude (K)**: Represents the maximum velocity of the star as it wobbles due to the gravitational influence of the planet.
    - **Orbital Period (P)**: The time taken by the planet to complete one orbit around the star.
    - **Mass of the Planet (M_p)**: The mass of the exoplanet, which influences the gravitational pull it exerts on the star.
    - **Mass of the Star (M_*)**: The mass of the host star, which also affects the radial velocity measurements.
    - **Eccentricity (e)**: A measure of how elliptical the orbit is, affecting the shape of the velocity curve.
    """)


with tab2:
    st.header("Planet Details")
    st.markdown("""
    This section displays details about the exoplanets, including:
    - **pl_name**: Name of the exoplanet
    - **hostname**: Host star name
    - **pl_bmasse**: Mass of the planet (in Earth masses)
    - **pl_orbper**: Orbital period (in days)
    - **pl_orbsmax**: Semi-major axis (in AU)
    - **st_mass**: Mass of the host star (in Solar masses)
    """)
    if df is not None:
        st.write("Displaying detailed information about planets:")
        st.dataframe(df[['pl_name', 'hostname', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax', 'st_mass']])
    

with tab3:
    st.header("3D Visualization of Planetary Orbits")
    st.markdown("""
    The 3D plot displays the relationship between the semi-major axis, orbital period, and planet mass of the detected exoplanets. 
    This visualization helps in understanding how different planets interact with their host stars based on their distances and masses.
    """)
    
    
    if df is not None:
        fig_3d = px.scatter_3d(df, x='pl_orbsmax', y='pl_orbper', z='pl_bmasse', color='pl_name',
                                labels={'pl_orbsmax': 'Semi-major Axis (AU)', 'pl_orbper': 'Orbital Period (days)', 'pl_bmasse': 'Planet Mass (Earth Masses)'})
        fig_3d.update_layout(title="Interactive visualization of Planetary Orbits")
        st.plotly_chart(fig_3d)
    st.markdown("Refer the link for an interactive visualisation: [EYES ON EXOPLANETS](https://eyes.nasa.gov/apps/exo/#/?browse_destination=Detection_Method)")   
        
with tab4:
    st.header("Habitable Zone Analysis")
    st.markdown(""" ### This tab identifies and analyzes exoplanets that lie within the habitable zones (HZ) of their respective stars.
    The HZ, also known as the "Goldilocks zone," is the region around a star where conditions may be just right for liquid water to exist on a planet's surface—crucial for life as we know it.
    
    ### Steps and Calculations
    1. **Data Fetching**: The data is fetched from the NASA Exoplanet Archive and includes parameters such as planet mass, orbital period, semi-major axis, stellar mass, and effective temperature. 
    2. **Habitable Zone Calculation**: The inner and outer boundaries of the HZ are calculated based on the star's effective temperature (`st_teff`). 
        The luminosity (`L`) of the star is calculated as: 
        $$ 
        L = \\left( \\frac{T_{star}}{T_{sun}} \\right)^4 
        $$ 
        Using the luminosity, the inner (`r_{inner}`) and outer (`r_{outer}`) boundaries of the HZ are given by: 
        $$ 
        r_{inner} = \\sqrt{\\frac{L}{S_{eff,sun}(0) + a(0) \\cdot (T_{star} - T_{sun}) + b(0) \\cdot (T_{star} - T_{sun})^2 + c(0) \\cdot (T_{star} - T_{sun})^3}}
        $$ 
        
        $$
        r_{outer} = \\sqrt{\\frac{L}{S_{eff,sun}(1) + a(1) \\cdot (T_{star} - T_{sun}) + b(1) \\cdot (T_{star} - T_{sun})^2 + c(1) \\cdot (T_{star} - T_{sun})^3}} 
        $$
        
    3. **Variables used**: 
        - `star_teff`: Effective temperature of the star. 
        - `T_sun`: Effective temperature of the sun (5778 K). 
        - `L`: Luminosity of the star in terms of solar luminosity. 
        - `S_eff_sun`: Effective stellar flux at the inner and outer edges of the HZ. 
        - `a, b, c`: Coefficients for the polynomial expression.
        
    4. **Identifying Exoplanets in the Habitable Zone**: 
        The semi-major axis (`pl_orbsmax`) of each exoplanet is compared against the calculated HZ boundaries to determine if it lies within the HZ.
    5. **Visualization**: A 3D scatter plot visualizes the exoplanets within their respective habitable zones, showing the relationship between the HZ boundaries and the exoplanet's orbital distance. 
    """)
    df = fetch_exoplanet_data(limit=dataset_count) 
    if df is not None:
         
        # Calculate Habitable Zone for each star
        habitable_zones = df['st_teff'].apply(calculate_habitable_zone)
        df['hz_inner'] = habitable_zones.apply(lambda x: x[0]) 
        df['hz_outer'] = habitable_zones.apply(lambda x: x[1])
        # Identify Exoplanets within the Habitable Zone 
        df['in_hz'] = df.apply(lambda row: row['hz_inner'] <= row['pl_orbsmax'] <= row['hz_outer'], axis=1) 
        # Filter Exoplanets in the Habitable Zone
        habitable_exoplanets = df[df['in_hz']]
        st.subheader("Exoplanets within the Habitable Zone") 
        st.write(habitable_exoplanets[['pl_name', 'hostname', 'pl_orbsmax', 'hz_inner', 'hz_outer']])
        # 3D Scatter Plot of Exoplanets within the Habitable Zone 
        fig = px.scatter_3d(habitable_exoplanets, x='hz_inner', y='hz_outer', z='pl_orbsmax', color='pl_name', 
                        labels={'hz_inner': 'HZ Inner Boundary (AU)', 'hz_outer': 'HZ Outer Boundary (AU)', 'pl_orbsmax': 'Orbital Distance (AU)'}, 
                        title='Exoplanets within the Habitable Zone') 
        st.plotly_chart(fig)
    st.subheader ("Habitable Zone Explanation Video")    
    st.video("https://youtu.be/J04YN9azln8?si=zIwLiVz7FeMRIxAl") 
    st.markdown("[Reference Paper for the analysis method](https://iopscience.iop.org/article/10.1088/2041-8205/787/2/L29/pdf)")
            
with tab5:
    st.header("Real-Time Data Updates")
    st.markdown(""" The **Refresh Data** allows users to fetch the most up-to-date exoplanet data from the NASA Exoplanet Archive. """) 
    if st.button('Refresh Data'):
        df = fetch_exoplanet_data(limit=dataset_count)
        st.write("Data refreshed successfully!")
        st.dataframe(df[['pl_name', 'hostname', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax', 'st_mass']])
    st.subheader (" Radial Velocity Method Explanation Video ")    
    st.video("https://youtu.be/rN7uuqLKv0I?si=L400cu8qcas3dVjU") 
    st.markdown("[Reference Paper for the project](https://arxiv.org/pdf/2404.09143)") 
    



# Groq AI response function with rate limiting
def get_ai_response(query: str) -> str:
    if not groq_client:
        return "⚠️ AI Assistant is currently unavailable. Please configure the GROQ_API_KEY."
    
    try:
        # Rate limiting: Max 1 request per 2 seconds
        import time as time_module  # Import with alias to avoid any conflicts
        current_time = time_module.time()
        time_since_last = current_time - st.session_state.last_request_time
        
        if time_since_last < 2:  # Wait at least 2 seconds between requests
            time_module.sleep(2 - time_since_last)
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in exoplanetary science. 
                    Provide clear, concise scientific explanations using established astronomical concepts 
                    and current research. Reference NASA's exoplanet database when relevant.
                    Keep responses under 500 words and format with proper markdown for readability."""
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            model="openai/gpt-oss-120b"  # Fast and accurate
            temperature=0.7,
            max_completion_tokens=1200,
            reasoning_effort="medium"
            top_p=0.9,
        )
        st.session_state.last_request_time = time_module.time()
        st.session_state.request_count += 1
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        
        # Handle rate limit errors gracefully
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return "⚠️ Too many requests. Please wait a moment and try again."
        
        # Handle other errors
        return f"❌ Error: {error_msg}\n\nPlease try again or rephrase your question."


with tab6:
    st.header("Ask AI About Exoplanets")
    
    # Display usage stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Asked (This Session)", st.session_state.request_count)
    with col2:
        if st.session_state.last_request_time > 0:
            import time as time_module
            seconds_ago = int(time_module.time() - st.session_state.last_request_time)
            st.metric("Last Question", f"{seconds_ago}s ago")
    
    # Example questions
    st.markdown("""
    ### Example questions:
    - What is the radial velocity method of detecting exoplanets?
    - How do scientists determine if an exoplanet is in the habitable zone?
    - What are hot Jupiters and why are they important?
    - How do transit observations help in detecting exoplanets?
    - What is the difference between super-Earths and mini-Neptunes?
    - How does eccentricity affect a planet's orbit?
    """)

    # Use form to allow Enter key submission
    with st.form(key="ai_question_form", clear_on_submit=False):
        user_query = st.text_input("Enter your question about exoplanets:", 
                                    placeholder="Ask anything about exoplanets...",
                                    key="question_input")
        
        submit_button = st.form_submit_button("Ask AI", type="primary")
    
    if submit_button:
        if user_query:
            with st.spinner('Generating response...'):
                response = get_ai_response(user_query)
                
                st.write("### AI Response:")
                st.markdown(response)
                
                # Add feedback buttons
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    st.button("👍 Helpful")
                with col2:
                    st.button("👎 Not helpful")
        else:
            st.warning("Please enter a question before submitting.")
    
    # Add disclaimer
    st.caption("Powered by Groq (Llama 3.3 70B) • Responses are AI-generated and should be verified with scientific sources.")

