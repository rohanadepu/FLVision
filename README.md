# FLVision

Faculty Advisors:

Graduate Mentors:

Undergraduate Researchers:

* Rohan Adepu

* Kevin Kostage

* Jenaya Monroe


## Steps to Run

* Clone the repo

* Open terminal in the DockerSetup directory

* Run with Docker: 
  * docker-compose up
  
* If there was an error, run:
  * docker build -t flwr-server -f Dockerfile.server .
  * docker build -t flwr-client -f Dockerfile.client .
  * docker-compose build
  * docker-compose up

* Run with AERPAW:
  * Ensure all files are downloaded and files are configured.
  * in clients go to DockerSetup and use 'python3 client.py'
  * in server go to DockerSetup and use 'python3 server.py'
