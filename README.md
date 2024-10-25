# HFL-DNN-GAN-NIDS

Faculty Advisors:

Graduate Mentors:

Undergraduate Researchers:

* Kevin Kostage

* Sean Peppers


## Steps to Run


* Clone the repo

* Run with Docker: 
  * Open terminal in the DockerSetup directory
  * docker-compose up
  
* If there was an error, run:
  * docker build -t flwr-server -f Dockerfile.server.
  * docker build -t flwr-client -f Dockerfile.client.
  * docker-compose build
  * docker-compose up

* Run with AERPAW:
  * Ensure all files are downloaded and files are configured.
    * Use Scp AERPAW_setup_Scrpit and run it in root to properly set up node for experiments.
  * in clients use 'python3 clientExperiment.py'
    * use "--dataset IOTBOTNET" after that statement to use the iotbotnet dataset
  * in server use 'python3 server.py'


## Client File Summary:
  * Imports / ENV set up
  * Script Arguments parsing
  * Data Load / Processing CICIOT
  * * Loading Settings
    * Feature and Label Mappings
    * Loading
      * Functions
      * Sampling Files & Train/Test Split
      * Extracting to Data from Sampled Files into Dataframe 
    * Processing
      * Feature Selection
      * Encoding
      * Normalizing
      * X y Split & assigned to model


  * Data Load / Processing IOTBOTNET
    * Feature Mappings
    * Loading
    * * Loading Settings 
      * Functions
      * Loading Specific Attack Data into Dataframes
      * Combine Specific Attacks into a Single Dataframe & Make it Binary Labels
    * Processing
      * Cleaning
      * Train Test Split
      * Feature Selection
      * Encoding
      * Normalizing
      * X y Split & assigned to model
  

  * Model Setup
    * Main Hyperparameters
    * CICIOT MODEL Layer structures
    * IOTBOTNET MODEL Layer Structures
    * Custom Optimizer for Differential Privacy
    * Default Optimizer & Compile Model
    * Callback Components
    * Model Analysis


  * Federated Learning Setup
    * Setting weights Def
    * Fitting Model Def
    * Evaluating Model Def


  * Start Client

  
