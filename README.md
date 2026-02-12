# gust-front-detection-webapp
This is a prototype web interface to interact with the gust front detection algorithm found [here](https://github.com/firelab/NFGDA).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Frontend](#frontend)
- [Backend](#backend)
- [Docker](#docker)
- [Deployment](#deployment)

# Prerequisites
How to setup dependencies for project. 

# Frontend

To build the frontend:

1. Clone this repository with `git clone`
2. Navigate to the frontend directory with `cd /gust-front-detection-webapp/frontend`
3. Install packages with `npm install`
4. Run frontend with `npm run dev`

The frontend is now accessible at `http://localhost:5173/`

# Backend
TODO: Add instructions on how to build and run the backend.

Backend directory structure: 
    - app.py contains the API endpoints
    - apis/ contains the API endpoint definitions
        - api definitions found here call logic defined in src/
    - src/ contains the backend logic
        - Not responsible for API endpoints that orchestrate or handle HTTP requests
        - Contains the business logic of the application only
        
# Docker
Describe how to containerize and run the project

# Deployment
Explain how the application is deployed
