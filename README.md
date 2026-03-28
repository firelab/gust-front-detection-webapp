## gust-front-detection-webapp

This is a prototype web interface to interact with the gust front detection algorithm found [here](https://github.com/firelab/NFGDA).

# How To Run

First time running project?

1. Navigate to project directory containing `docker-compose.yml`
2. Run `docker compose up -d --build`
3. Navigate to http://localhost:5173
4. Play widdit

- To re-launch app, run `docker compose up -d`
- To restart docker containers, run `docker compose restart -d`

# Frontend

# Backend (/backend and /nfgda_service)

Backend directory structure:

- app.py contains the API endpoints
- /apis contains the API endpoint definitions
- API call logic defined in src/
- src/ contains the backend logic (Not responsible for API endpoints that orchestrate or handle HTTP requests. Contains the business logic of the application only)

NFGDA Service directory structure:

- /nfgda_service contains the NFGDA service logic
- responsible for all NFGDA execution, output processing, and file management

# Todo (before MSU handoff)

- Guard against short jobs that run forever for some reason
- Figure out zoom level / blank frame issue on frontend
- Switching to a new station view pauses slide deck playthrough
- Convert geotiff output to cloud-optimized-geotiffs
- Code cleanup / add comments where necessary

# "Nice to have" features

- Average time to job completion estimator (small addition: new counter in redis, average out)
- Serve tiles instead of individual GeoTIFFs (big refactor)
- Hash job IDs to make them unguessable, so resources can't be directly accessed via URL (little development effort, likely med/large refactor effort)
