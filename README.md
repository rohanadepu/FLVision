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


## Client File Summary:
  * Data Load / Processing CICIOT
    * Loading
      * Sampling Files & Train/Test Split
      * Extracting to Dataframe
    * Processing
      * Remapping & Feature Selection
      * Encoding
      * Normalizing
      * X y Split


  * Data Load / Processing IOTBOTNET
    * Loading
      * Functions
      * Loading Specific Attack Data into Dataframes
      * Combine Specific Attacks into a Single Dataframe & Make it Binary Labels
      * Train Test Split
    * Processing
      * Feature Selection
      * Cleaning
      * Encoding
      * Normalizing
      * X y Split
  

  * Model Setup
    * Hyperparameters
    * CICIOT MODEL
    * IOTBOTNET MODEL
    * Custom Optimizer for Differential Privacy
    * Compile Model
    * Callback Components
    * Model Analysis


  * Federated Learning Setup
    * Setting weights Def
    * Fitting Model Def
    * Evaluating Model Def


  * Start Client

  
