# Exoplanet Detection Simulator

This project provides a simulation environment for exoplanet detection techniques, implemented using Python and Streamlit. It helps users understand and visualize various methods used in detecting exoplanets.

## Features

- Interactive simulation of exoplanet detection methods
- Visualization of transit light curves
- Data analysis tools for exoplanet signals
- Real-time parameter adjustment
- Educational insights into exoplanet detection techniques

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/exoplanet-detection-sim.git
cd exoplanet-detection-sim
```

2. Run the setup script:
```bash
# On Unix/Linux/MacOS:
chmod +x setup.sh
./setup.sh

# On Windows:
# Run setup.sh using Git Bash or WSL
```

3. Create a `.streamlit/secrets.toml` file with your configuration (see Configuration section)

## Configuration

Create a `.streamlit/secrets.toml` file with the following structure:
```toml
[nasa_api]
api_key = "your-nasa-api-key"

[simulation]
default_star_mass = 1.0
default_planet_radius = 1.0
default_orbital_period = 365.0
```

## Usage

Run the Streamlit app:
```bash
streamlit run Exoplanet.py
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
