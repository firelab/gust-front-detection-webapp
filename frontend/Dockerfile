
FROM node:20
#set working directory inside container
WORKDIR /app
#copy requirements 
COPY package*.json ./
#install dependancies
RUN npm install
#copy front end into container
COPY . .
#Expose port 5173
EXPOSE 5173
#command to run webpage
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]