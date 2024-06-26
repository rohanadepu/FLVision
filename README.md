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
  * in clients use 'python3 client.py'
  * in server use 'python3 server.py'


Client File Summary:
  * Data Load / Processing CICIOT
    * Loading
      * Sampling Files
      * Processing

  * Data Load / Processing IOTBOTNET
    * Loading
      * Sampling Files
      * Processing
  
  * Model Setup
    * Hyperparameters
    * CICIOT MODEL
    * IOTBOTNET MODEL
    * Differential Privacy
    * Compile Model
    * Callback Components
    * Model Analysis

  * Federated Learning Setup
    * Setting weights Def
    * Fitting Model Def
    * Evaluating Model Def

  * Start Client

  
