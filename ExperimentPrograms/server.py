#########################################################
#    Imports / Env setup                                #
#########################################################

import argparse
import flwr as fl

#########################################################
#    Script Args                                        #
#########################################################

parser = argparse.ArgumentParser(description='Select to enabled sec agg')
parser.add_argument('--secAggP', action='store_true', help='Enable model adversarial training')

args = parser.parse_args()

secAggPlusEnabled = args.secAggP

if secAggPlusEnabled:
    print("Secure Aggregation Plus Enabled \n")
else:
    print("Secure Aggregation Plus Disabled \n")
#########################################################
#    Setup Server                  #
#########################################################

if secAggPlusEnabled:

    # Define the server configuration with SecAgg+ enabled
    secagg_config = fl.server.SecAggConfig(
        secagg_type="SecAgg+",  # Enable SecAgg+
        threshold_ratio=0.5,  # Define the threshold ratio for SecAgg+
    )

    # Start the server with the defined configuration
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=10, secagg_config=secagg_config)
    )


#########################################################
#    Initialize Server                  #
#########################################################
else:
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=10)
    )
