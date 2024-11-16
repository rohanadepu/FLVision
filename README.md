# HFL-DNN-GAN-NIDS

## Faculty Advisors:

**Dr. Qu, Chenqi**

**Dr. Prasad, Calyam**

**Dr. Reshmi Mitra**


## Graduate Mentors:

* **Mogollon, Juan:** Computer Vision Specialist, AI Engineer

* **Haughton, Trevontae:** AERPAW Test-Bed Specialist, FlyPaw Manager, End-to-End Engineer


## Undergraduate Researchers:

* **Kevin Kostage:** Deep Learning Engineer, End-to-End Engineer, Cloud-Network Specialist

* **Paulo Drefahl:** Network Engineer, Cyber-Security Specialist, Full-Stack Developer

* **Sean Peppers:** Deep Learning Engineer, Cyber-Attack Specialist, Experiment Tech.

* **Rohan Adepu:** Cyber-Attack Specialist, Experiment Tech.

* **Jenaya Monroe:** Drone Hardware Specialist, Test-Bed Manager



## Previous Works
_**Enhancing Autonomous Intrusion Detection System with Generative Adversarial Networks:**_ https://ieeexplore.ieee.org/document/10678662

**_Enhancing Drone Video Analytics Security Management using an AERPAW Testbed:_** https://ieeexplore.ieee.org/document/10620812

## Steps to Run

* **Clone the repo**

* **Run Test Demo with Docker:** 
  * Open terminal in the _DockerSetup_ directory
    *       docker-compose up
  
  * If there was an error, run:
    *       docker build -t flwr-server -f Dockerfile.server.
    
            docker build -t flwr-client -f Dockerfile.client.
    
            docker-compose build
    
            docker-compose up

    
* **Run with AERPAW:**
  * Connect to TestBed:
  
  * Once Connected to the Portable-Server Client Nodes: Go to _FLVision/hflClient_ directory run any of the Client files to initiate the devices to connect to the host server to initiate training 
    *       python3 [name of program]
      * use "--dataset IOTBOTNET" after that statement to use the iotbotnet dataset, if not the script will use the CICIOT Dataset.
  

* Once Connected to the Fixed-Server Host nodes: Go to _FLVision/hflServer/_ to run the various server scripts.
  * To Run the basic server script with no server-side saving.
    *     python3 serverBase.py


## Client File Process Summary:
  * **Imports / ENV set up**
  * **Script Arguments parsing**
  * **Data Load / Processing CICIOT**
  * * Loading Arguments
    * Feature and Label Mappings
    * Loading
      * Helper Functions
      * Sampling Files & Train/Test Split
      * Extracting to Data from Sampled Files into respective Dataframes
    * Processing
      * Feature Selection
      * Encoding
      * Normalizing
      * X y Split & assigned to model


  * **Data Load / Processing IOTBOTNET**
    * Feature Mappings
    * Loading
    * * Loading Arguments 
      * Helper Functions
      * Loading Specific Attack Data into Dataframes
      * Combine Specific Attacks into a Single Dataframe & Make it Binary Labels
    * Processing
      * Cleaning
      * Train Test Split
      * Feature Selection
      * Encoding
      * Normalizing
      * X y Split & assigned to model
  

  * **NIDS Model Setup**
    * Main Hyperparameter defined
    * CICIOT MODEL Layer structure defined
    * IOTBOTN MODEL Layer Structure defined
    * Default Optimizer & Compile Model
    * Model Analysis


  * **Discriminator Model Setup**
    * Main Hyperparameters defined
    * MODEL Layer Structure defined
    * Default Optimizer & Compile Model
    * Model Analysis


  * **Generator Model Setup**
    * Main Hyperparameters defined
    * MODEL Layer structure defined
    * Default Optimizer & Compile Model
    * Model Analysis


  * **GAN Model Setup**
    * Main Hyperparameters defined
    * Combine defined DISC. & GEN. MODEL Layer structures
    * Default Optimizer & Compile Model
    * Model Analysis
    
    
  * **Initiate Client**


  * **Federated Learning Setup**
    * Client connects to host server to start training process
    * Setting param Def
    * Fitting Model Def
    * Evaluating Model Def
    * Distributes Aggregated Model Back to Clients


  

  
