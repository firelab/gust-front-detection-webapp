# Gust Front Web App

This is a prototype web interface for the [Neuro-Fuzzy Gust Front Detection Algorithm](https://github.com/firelab/NFGDA).

## How To Run

First time running the project?

1. Clone this repo and navigate to project directory containing `docker-compose.yml`
2. Run `docker compose up -d --build`
3. Open http://localhost:5173 in your browser

- See the browser console for additional logs.
- To re-launch app, run `docker compose up -d`
- To restart docker containers, run `docker compose restart -d`

## Frontend

The frontend is a React app built using Vite. `frontend/src/` contains all the logic. The top-level component is `App.jsx`.

## Backend (/backend and /nfgda_service)

Backend directory structure:

- `app.py` contains the API endpoints
- `/apis` contains the API endpoint definitions
- API call logic is defined in `src/`
- `src/` contains the backend logic (Not responsible for API endpoints that orchestrate or handle HTTP requests. Contains the business logic of the application only)

NFGDA Service directory structure:

- `/nfgda_service` contains the NFGDA service logic
- `/nfgda_service/algorithm` contains the original NFGDA code and script with some slight tweaks. Responsible for all NFGDA execution, output processing, and file management

And then there's a redis instance living at port 6379 where all the job status and asset information is stored.

## Todo

### Todo before MSU handoff

- [ ] Add a "clear" button to the map that clears all job assets from the map
- [ ] Discuss pixel-width of gust fronts written to output file next team meeting
- [ ] Diff the NFGDA code used in nfgda_service with the original NFGDA code, see if there are any useful features we're missing out on or bugs we introduced

### "Nice to have" features

- There a should probably be a warning that shows up for small numbers of assets per job (2 frames produced or less). Maybe if not enough assets are produced, the job request could automatically re-run with a larger time window?
- Average time to job completion estimator (small addition: new counter in redis, average out)
- Serve tiles instead of individual GeoTIFFs (big refactor, honestly might not be worth at as Cloud-optimized-geotiffs are kinda the future anyway)
- Hash job IDs to make them unguessable, so resources can't be directly accessed via URL (little development effort, likely med/large refactor effort)

### Todo after MSU handoff (futures devs read this pls)

- Check that automatic asset deletion occurs within the timeframe specified (should be 24 hours)
- Familiarize with the .env file and environment variables, and what they do

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 ROCKY MOUNTAIN RESEARCH STATION (RMRS)
MISSOULA FIRE SCIENCES LABORATORY

  
