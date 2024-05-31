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

* Run: 
  * docker-compose up
  
* If there was an error, run:
  * docker build -t flwr-server -f Dockerfile.server .
  * docker build -t flwr-client -f Dockerfile.client .
  * docker-compose build
  * docker-compose up  
