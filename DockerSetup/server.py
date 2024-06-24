#########################################################
#    Imports / Env setup                                #
#########################################################

import flwr as fl

#########################################################
#    Initialize, Setup, & Start Server                  #
#########################################################

fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=10)
)
