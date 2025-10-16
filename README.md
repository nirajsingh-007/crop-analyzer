# Crop Analyzer - Satellite-Based Crop Health Monitoring

---

## Introduction

**Crop Analyzer** is a **Satellite-Based Crop Health Monitoring System** that leverages satellite imagery and vegetation indices to detect and visualize the health status of crops.  
It helps farmers, researchers, and agricultural institutions monitor crop growth, detect stress or diseases early, and make data-driven decisions for improved productivity.

---

## Key Features

- Integration with satellite imagery (Sentinel)
- NDVI (Normalized Difference Vegetation Index) computation
- Visualization of healthy vs unhealthy crop zones
- Interactive map-based UI using Leaflet
- Backend powered by Python (Flask)
- Frontend built with React + TypeScript + Vite
- JSON-based crop health data integration (Crop based NDVI Range)
- Detection of abnormal vegetation health conditions

---

## Installation & Setup

### Prerequisites

- Python 3.8 or later  
- Node.js and npm (or yarn)  
- Git  

---

### Clone the repository
```sh
git clone https://github.com/nirajsingh-007/crop-analyzer.git
cd crop-analyzer/server
```

### Create virtual environment
```sh
python -m venv venv
source venv/bin/activate
```
(on Windows: venv\Scripts\activate)

### Install dependencies
```sh
pip install -r requirements.txt
```

### Run backend server
```sh
python server.py
```

### Frontend Setup (in different terminal window)
```sh
npm install
npm run dev
```
Add your google-cloud-project-id in .env file (update the .env.example file to .env)

Now open your browser and visit:
http://localhost:5173/ (or whichever port Vite shows)

## Acknowledgement
- European Space Agency (ESA) - Sentinel satellite data
- Inspiration from Vedas Space Application Center, ISRO (https://vedas.sac.gov.in/)

---

## Our Team
- Niraj Singh (Team Leader) - https://github.com/nirajsingh-007
- Rahul Varma - https://github.com/Meruem09
- Mohit Kamble - https://github.com/Theshmphony7580
- Shyam Jadav
- Nirav Chaudhari
