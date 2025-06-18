# Exoplanet Detection Simulation using Radial Velocity Method
# This code fetches data from the NASA Exoplanet Archive, calculates radial velocity curves,
# visualizes the data in 3D, and allows users to interact with the data using Streamlit.
# Required Libraries
# pip install streamlit plotly pandas numpy requests astropy google-generativeai

# Import necessary libraries   
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from astropy.constants import G
from astropy import units as u
import google.generativeai as genai
import time

# Configure the Gemini API
genai.configure(api_key=st.secrets["gemini_api"]) 
# Initialize the model with the correct name
model = genai.GenerativeModel('gemini-2.0-flash-001')  

# Step 1: Fetch Exoplanet Data from NASA Exoplanet Archive
@st.cache_data
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
        pl_bmasse > 0 AND
        pl_orbper > 0 AND
        pl_orbsmax > 0 AND
        st_mass > 0
    ORDER BY
        pl_orbper ASC
    """
    params = {
        "query": query,
        "format": "json"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        return df
    else:
        st.error("Error fetching data from NASA Exoplanet Archive.")
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
    time = np.linspace(0, time_span, 1000)  # Time points (days)
    velocity = K * np.sin(2 * np.pi * time / P)  # Radial velocity at each time point
    return time, velocity
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
    r_inner = np.sqrt(L / (S_eff_sun[0] + a[0] * (T_star - T_sun) + b[0] * (T_star - T_sun)**2 + c[0] * (T_star - T_sun)**3))
    r_outer = np.sqrt(L / (S_eff_sun[1] + a[1] * (T_star - T_sun) + b[1] * (T_star - T_sun)**2 + c[1] * (T_star - T_sun)**3)) 
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
                time, velocity = generate_radial_velocity_curve(K, orbital_period, time_span)

                # Add the curve to the Plotly figure
                fig.add_trace(go.Scatter(x=time, y=velocity, mode='lines', name=f'{planet_name} ({star_name})'))

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
    

@st.cache_data(ttl=3600)  # Cache responses for 1 hour
def get_gemini_response(prompt):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Create a more focused prompt with context
            full_prompt = f"""As an expert in exoplanetary science, please answer the following question:
            {prompt}
            
            Please provide a clear, scientific explanation using established astronomical concepts and current research.
            If relevant, include references to NASA's exoplanet database or recent discoveries."""
            
            # Set generation config for better responses
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=40,
                max_output_tokens=208,
            )
            
            # Generate the response with the configured settings
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )
            
            return response.text
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            raise e

with tab6:
    st.header("Ask Gemini Anything About Exoplanets")
    
    # Add some example questions
    st.markdown("""
    ### Example questions you can ask:
    - What is the radial velocity method of detecting exoplanets?
    - How do scientists determine if an exoplanet is in the habitable zone?
    - What are hot Jupiters and why are they important?
    - How do transit observations help in detecting exoplanets?
    """)

    user_prompt = st.text_input("Enter your question about exoplanets:")

    if st.button("Ask Gemini"):
        if user_prompt:
            try:
                with st.spinner('Generating response...'):
                    response = get_gemini_response(user_prompt)
                    
                if response:
                    st.write("Gemini's Response:")
                    st.markdown(response)
                else:
                    st.error("No response generated. Please try again with a different question.")
            
            except Exception as e:
                st.error(f"An error occurred while getting the response. Please try again in a moment. Error: {str(e)}")
                st.info("If the error persists, try refreshing the page or asking a different question.")
        else:
            st.warning("Please enter a question before submitting.")



