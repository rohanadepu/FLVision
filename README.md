# FLVision

Faculty Advisors:

Graduate Mentors:

Undergraduate Researchers:

* Rohan Adepu

* Kevin Kostage

* Jenaya Monroe


## Steps to Run

* Clone the repo

* open terminal in the DockerSetup directory

* enter: 
  * docker-compose up
  
* If this doesn't work enter:
  * docker build -t flwr-server -f Dockerfile.server .
  * docker build -t flwr-client -f Dockerfile.client .
  * docker-compose build
  * docker-compose up  
